//! Provides a less unsafe way to store multiple pointers to an object without
//! overhead - a middle ground between raw pointers and the `rental` crate.
//!
//! A common pattern in C and C++ code is storing multiple pointers to the same
//! (heap-allocated) object in different data structures to allow multiple ways
//! to look up the object. This is a pattern that maps relatively badly to Rust,
//! since it's difficult to track the different references. Rust code usually
//! has these options:
//!
//! * Store `Rc`s instead of pointers to the objects.
//!
//!   This makes it entirely safe, but incurs a (small) performance and memory
//!   overhead that might not be acceptable for the application.
//!
//! * Store indices into a `Vec` that stores the actual objects.
//!
//!   When storing array indices instead of `Rc`s or raw pointers, you get both
//!   safety (the worst that can happen is that a wrong value is returned or
//!   that the program panics when the index is out of bounds) and possibly
//!   better performance than `Rc`-based code, since no reference count needs to
//!   be maintained.
//!
//!   This approach works well for many applications, but also comes at a cost:
//!   The extra allocation done by the `Vec`, as well as complexities when
//!   removing values (do you replace it with an invalid object or move another
//!   object into its place?). It also means that you have to compute the
//!   object's address inside the `Vec`'s storage when accessing it, something
//!   that's not necessary when storing references or pointers.
//!
//! * Use raw pointers and manually ensure the usage is safe.
//!
//!   This works as well as it does in C and C++, but also relies on the
//!   programmer to write safe code and doesn't utilize Rust's type system and
//!   safety features at all.
//!
//! * Use the [`rental`](https://crates.io/crates/rental) crate to allow
//!   self-referencing structs.
//!
//!   The `rental` crate allows storing a reference in the same struct as the
//!   owning container in completely safe Rust, but has a very complex API and
//!   requires writing the Rust struct in question inside the `rental!` macro.
//!   The resulting code often looks vastly different from usual Rust code,
//!   having to jump through hoops and closures to access fields and having to
//!   put fields of the defined structs in `Box`es that should be unnecessary.
//!
//!   The `unsafe_box` crate has a vastly simpler interface compared to `rental`
//!   and provides simple, ordinary Rust types instead of macro magic (which
//!   means that it doesn't impose any restrictions on the code you can write),
//!   but at the cost of relying a bit on the programmer for correctness.
//!
//! This crate is meant to be used as a better alternative to the raw pointer
//! approach, while being simpler and easier to use than the `rental` crate.
//! While it doesn't try to make it entirely safe, it does employ a few
//! techniques to make writing that code in a safe way much easier:
//!
//! It separates the pointers to the object into one *owning pointer* of type
//! `UnsafeBox` that will destroy the object when it is dropped, and any number
//! of *referencing pointers* of type `UnsafeRef` that point to the object and
//! must be destroyed before the owning pointer is dropped (since that would
//! create a dangling pointer).
//!
//! In order to eliminate overhead, this is *not* generally checked, so the
//! burden of safety is still on the programmer using this library. However, in
//! debug mode, additional runtime checks are enabled that will panic when an
//! owning pointer is attempted to be freed when there are still referencing
//! pointers to the object.
//!
//! To uphold Rust's aliasing guarantees it is also important not to create a
//! mutable reference to the value from *any* existing `UnsafeRef` *or* the
//! `UnsafeBox` itself as long as any other reference exists. This is also
//! checked in debug mode, in a similar fashion to how `RefCell` works.
//!
//! The end result is that this approach should be exactly as fast as wrangling
//! raw pointers by hand (in release mode), but offers quite a decent level of
//! safety in debug mode (in debug mode, it basically acts like an
//! `Rc<RefCell<T>>` would).
//!
//! # Examples
//!
//! Basic usage:
//!
//! ```
//! # use unsafe_box::*;
//! let mut owner = UnsafeBox::new(vec![0, 1, 2]);
//! let mut r: UnsafeRef<_> = owner.create_ref();
//!
//! // Accessing the value requires `unsafe`. This access is safe since `owner`
//! // is still alive and no mutable reference exists.
//! assert_eq!(unsafe { &*r.unsafe_deref() }, &[0, 1, 2]);
//!
//! // Mutation is only allowed if no other references exist that were acquired
//! // with `unsafe_deref_*` (and, of course, `owner` must still be alive).
//! unsafe {
//!     r.unsafe_deref_mut()[0] = -123;
//! }
//!
//! // `UnsafeBox` supports the same `unsafe_deref_*` methods as `UnsafeRef`.
//! assert_eq!(unsafe { &*owner.unsafe_deref() }, &[-123, 1, 2]);
//! ```
//!
//! A more elaborate example, showing how to use `unsafe_box` to build a data
//! structure, is located in `examples/multi-lookup-storage.rs`.

#![doc(html_root_url = "https://docs.rs/unsafe-box/0.1.0")]
#![warn(missing_debug_implementations)]
#![warn(missing_docs)]

use std::{fmt, u32};
use std::ops::{Deref, DerefMut};
use std::cell::UnsafeCell;
use std::ptr::NonNull;

#[cfg(debug_assertions)]
use std::cell::Cell;    // used by the debug mode counters

/// A `Box<T>` with extra sauce.
///
/// An `UnsafeBox` allocates a value of type `T` on the heap, just like `Box`
/// (and also guaranteeing a stable address). When dropped, the value will be
/// deallocated, just like `Box` would.
///
/// `UnsafeBox` allows lightweight `UnsafeRef`s to be created that point to
/// the stored value, but do not manage its storage. For all intents and
/// purposes, `UnsafeRef` is just a raw pointer where *you*, the user, have to
/// make sure it isn't dropped before the corresponding `UnsafeBox`, and also
/// have to uphold Rust's aliasing guarantees. However, when debug assertions
/// are enabled, `UnsafeBox` will check that no `UnsafeRef`s are around when it
/// is dropped (which would create dangling pointers).
///
/// It is recommended to exhaustively test an application with enabled debug
/// assertions before turning them off for a release build.
///
/// Note that neither `UnsafeBox` nor `UnsafeRef` are `Send` or `Sync`. If you
/// need to do unsafe things like that in multithreaded code, only god (and
/// `Arc`) can help you.
pub struct UnsafeBox<T> {
    /// Pointer to the interior (shared with any `UnsafeRef`s that are created).
    ///
    /// Due to the shared nature of the `Inner<T>`, calling `NonNull::as_mut` is
    /// generally always unsafe, while `as_ref` is fine as long as aliasing of
    /// the value is handled properly (ie. no reference to the value is created
    /// unless the counts are checked or the user manually validates this as
    /// safe).
    ptr: NonNull<Inner<T>>,
}

impl<T> UnsafeBox<T> {
    /// Creates a new `UnsafeBox` owning `value`.
    ///
    /// This will do the same thing `Box::new` does - in particular, `value`
    /// will be moved into a heap allocation whose address is fixed even when
    /// the `UnsafeBox` is moved.
    ///
    /// The owned value will be freed when the returned `UnsafeBox` is dropped.
    ///
    /// # Examples
    ///
    /// ```
    /// # use unsafe_box::*;
    /// let my_box = UnsafeBox::new(vec![1, 2, 3]);
    ///
    /// // Value is alive as long as `my_box` is alive
    ///
    /// drop(my_box);   // Drops the box and the contained value
    /// ```
    #[inline]
    pub fn new(value: T) -> Self {
        let raw = Box::into_raw(Box::new(Inner::new(value)));
        // `Box::into_raw` should never return a null ptr, but the docs don't
        // specify this.
        let ptr = NonNull::new(raw).expect("Box::into_raw returned a NULL pointer");

        Self { ptr }
    }

    /// Unsafely dereference `self` to obtain immutable access to `T`.
    ///
    /// # Safety
    ///
    /// This is only safe to call if no mutable reference (be it in the form of
    /// a `RefMut` or a `&mut T`) to the stored value currently exists.
    // These docs are similar to the ones at `UnsafeRef::unsafe_deref`
    #[inline]
    pub unsafe fn unsafe_deref<'a>(&'a self) -> Ref<'a, T> {
        Ref::new(self.ptr.as_ref())
    }

    /// Unsafely dereference `self` to obtain mutable access to `T`.
    ///
    /// # Safety
    ///
    /// This is only safe to call if no other reference (neither `Ref` nor
    /// `RefMut`) to the stored value exists.
    // These docs are similar to the ones at `UnsafeRef::unsafe_deref_mut`
    #[inline]
    pub unsafe fn unsafe_deref_mut<'a>(&'a mut self) -> RefMut<'a, T> {
        RefMut::new(self.ptr.as_ref())
    }

    /// Create a new `UnsafeRef` pointing to the contained value.
    ///
    /// This requires mutable access to `self` and is expected to only be done
    /// when the `UnsafeBox` is first created.
    ///
    /// # Safety / Panics
    ///
    /// This method in itself is always save to call, but do note that the
    /// returned `UnsafeRef` does not contain a lifetime linking it to `self`,
    /// even though it is logically borrowing from `self`. This means that the
    /// Rust compiler cannot prevent you from dropping the owning `UnsafeBox`
    /// while an `UnsafeRef` is still around. This is done to allow storing it
    /// in the same struct as the original `UnsafeBox` without having to do the
    /// same gymnastics that the `rental` crate has to do.
    ///
    /// *You* have to ensure that the `UnsafeRef` returned by this method is
    /// dropped before the owning `UnsafeBox` is. When compiling with debug
    /// assertions (eg. using `cargo test`), this will be checked at runtime,
    /// causing a panic when attempting to drop an `UnsafeBox` that still has
    /// `UnsafeRef`s pointing to its value.
    #[inline]
    pub fn create_ref(&mut self) -> UnsafeRef<T> {
        // Safe: Uses `Cell` and immutable refs only
        unsafe {
            self.ptr.as_ref().inc_refs();
        }

        UnsafeRef { ptr: self.ptr }
    }
}

impl<T> Drop for UnsafeBox<T> {
    #[inline]
    fn drop(&mut self) {
        // `UnsafeBox` must be dropped after all referencing `UnsafeRef`s are
        // dropped.
        let ref_count = unsafe { self.ptr.as_ref().ref_count() };
        assert_eq!(ref_count, 0, "UnsafeBox dropped while there are still {} UnsafeRefs around", ref_count);
        // When the assert fails, the value is leaked. This is perhaps for the
        // better, since the other refs might try to access it when owning
        // structs are dropped.

        // There also must not be any immutable/mutable references to the value
        // around.
        // This is actually impossible, because the `UnsafeBox` cannot be
        // dropped while it's still borrowed, so this would require another
        // `UnsafeRef`, which in turn would trip the assertion above ^, so this
        // is just an internal assert.
        unsafe { self.ptr.as_ref().assert_no_refs() };

        unsafe {
            Box::from_raw(self.ptr.as_ptr());
        }
    }
}

/// The `Debug` impl of `UnsafeBox` will not access the interior value, but only
/// print its address.
impl<T> fmt::Debug for UnsafeBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("UnsafeBox").field(&self.ptr).finish()
    }
}

/// A non-owning reference to the value in an `UnsafeBox`.
///
/// An `UnsafeRef` can be created using [`UnsafeBox::create_ref`]. It must be
/// destroyed before the `UnsafeBox` is dropped. In debug mode (when debug
/// assertions are enabled), attempting to drop an `UnsafeBox` that still has an
/// `UnsafeRef` pointing to it will cause a panic.
///
/// [`UnsafeBox::create_ref`]: struct.UnsafeBox.html#method.create_ref
pub struct UnsafeRef<T> {
    ptr: NonNull<Inner<T>>,
}

impl<T> UnsafeRef<T> {
    /// Unsafely dereference `self` to obtain immutable access to `T`.
    ///
    /// # Safety
    ///
    /// This is only safe to call if the `UnsafeBox` owning the backing storage
    /// still exists and no mutable reference to the stored value currently
    /// exists (be it in the form of a [`RefMut`] or a `&mut T`).
    ///
    /// [`RefMut`]: struct.RefMut.html
    // These docs are similar to the ones at `UnsafeBox::unsafe_deref`
    #[inline]
    pub unsafe fn unsafe_deref<'a>(&'a self) -> Ref<'a, T> {
        Ref::new(self.ptr.as_ref())
    }

    /// Unsafely dereference `self` to obtain mutable access to `T`.
    ///
    /// # Safety
    ///
    /// This is only safe to call if the `UnsafeBox` owning the backing storage
    /// still exists and no other reference to the stored value exists (be it in
    /// the form of a [`Ref`], [`RefMut`], `&T` or `&mut T`).
    ///
    /// [`Ref`]: struct.Ref.html
    /// [`RefMut`]: struct.RefMut.html
    // These docs are similar to the ones at `UnsafeBox::unsafe_deref_mut`
    #[inline]
    pub unsafe fn unsafe_deref_mut<'a>(&'a mut self) -> RefMut<'a, T> {
        RefMut::new(self.ptr.as_ref())
    }
}

impl<T> Drop for UnsafeRef<T> {
    #[inline]
    fn drop(&mut self) {
        // Safe: Uses `Cell` and immutable refs only
        unsafe {
            self.ptr.as_ref().dec_refs();
        }
    }
}

/// The `Debug` impl of `UnsafeRef` will not access the interior value, but only
/// print its address.
impl<T> fmt::Debug for UnsafeRef<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("UnsafeRef").field(&self.ptr).finish()
    }
}

/// The inner value allocated on the heap using `Box`.
///
/// Note how every member is enclosed in its own cell type. This is because
/// we need to obtain immutable and mutable access to the counters while a
/// (possibly mutable) reference to the value exists and `UnsafeCell` requires
/// that any created mutable references are unique (this is Rust's underlying
/// model, after all), so we can't just put `Inner` into a single `UnsafeCell`.
struct Inner<T> {
    /// The actual value, stored in an `UnsafeCell` because the compiler no
    /// longer really has any aliasing information.
    ///
    /// This is a slight optimization regression compared to Rust references,
    /// but C/C++ technically have the same thing (LLVM does alias analysis
    /// anyways).
    value: UnsafeCell<T>,
    /// Number of active `UnsafeRef`s pointing to the value.
    ///
    /// Does not include the initial `UnsafeBox` (so it starts out as 0).
    #[cfg(debug_assertions)]
    ref_count: Cell<u32>,
    /// Total number and type of references obtained from the `UnsafeBox` or
    /// `UnsafeRef`s.
    ///
    /// If this is `u32::MAX`, a mutable reference to the value exists and
    /// further attempts to obtain any kind of reference to the value are
    /// illegal and will panic in debug mode and may cause UB in release mode.
    ///
    /// If this is 0, no references exist and it's okay to obtain either
    /// immutable references or a mutable reference.
    ///
    /// If this is any other value, that number of immutable references exist
    /// and it's illegal to obtain a mutable reference, but more immutable
    /// references may be created.
    #[cfg(debug_assertions)]
    borrows: Cell<u32>,
}

#[cfg_attr(not(debug_assertions), allow(unused))]
impl<T> Inner<T> {
    fn new(value: T) -> Self {
        Self {
            value: UnsafeCell::new(value),
            #[cfg(debug_assertions)]
            ref_count: Cell::new(0),
            #[cfg(debug_assertions)]
            borrows: Cell::new(0),
        }
    }

    #[cfg(debug_assertions)]
    fn ref_count(&self) -> u32 {
        self.ref_count.get()
    }

    #[cfg(not(debug_assertions))]
    fn ref_count(&self) -> u32 {
        0
    }

    fn inc_refs(&self) {
        #[cfg(debug_assertions)]
        {
            self.ref_count.set(self.ref_count.get() + 1);
        }
    }

    fn dec_refs(&self) {
        #[cfg(debug_assertions)]
        {
            self.ref_count.set(self.ref_count.get() - 1);
        }
    }

    fn obtain_imm_ref(&self) {
        #[cfg(debug_assertions)]
        {
            assert!(self.borrows.get() < u32::MAX, "cannot borrow value immutably when it's already borrowed mutably");
            self.borrows.set(self.borrows.get() + 1);
        }
    }

    fn release_imm_ref(&self) {
        #[cfg(debug_assertions)]
        {
            self.borrows.set(self.borrows.get() - 1);
        }
    }

    fn obtain_mut_ref(&self) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.borrows.get(), 0, "cannot borrow value mutably when it's already borrowed");
            self.borrows.set(u32::MAX);
        }
    }

    fn release_mut_ref(&self) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.borrows.get(), u32::MAX, "internal error: ref count no longer u32::MAX even with mutable ref around (count: {})", self.borrows.get());
            self.borrows.set(0);
        }
    }

    fn assert_no_refs(&self) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.borrows.get(), 0, "internal error: attempt to drop UnsafeBox while value is still borrowed");
        }
    }
}

/// An immutable reference to a `T`.
///
/// This is basically the same as a `&'a T`, but can do additional safety checks
/// when debug assertions are enabled.
///
/// # Safety
///
/// As long as an instance of `Ref` exists, no mutable references to the same
/// value may be created. Creating additional `Ref`s (which are immutable) to
/// the same value is fine. This is the same rule that Rust's native references
/// follow.
pub struct Ref<'a, T: 'a> {
    ptr: &'a T,
    #[cfg(debug_assertions)]
    inner: &'a Inner<T>,
}

impl<'a, T: 'a> Ref<'a, T> {
    unsafe fn new(ptr: &'a Inner<T>) -> Self {
        ptr.obtain_imm_ref();
        Self {
            ptr: &*ptr.value.get(),
            #[cfg(debug_assertions)]
            inner: ptr,
        }
    }

    /// Convert this `Ref` into a normal Rust reference with lifetime `'a`.
    ///
    /// This will **disable** the debug mode borrow conflict check for this
    /// reference, but may be required when returning a mutable reference to
    /// some outside code. It will also **disable** the dangling pointer check
    /// for the returned reference, which means that you do not get a panic when
    /// you destroy all `UnsafeRef`s and then the `UnsafeBox` while there's
    /// still an unchecked Rust reference around.
    ///
    /// # Safety
    ///
    /// This function is generally safe to call if the invariants of the `Ref`
    /// are upheld, which you have to do anyways, so this does not have to be
    /// marked as an unsafe function.
    // These docs are similar to the ones at `RefMut::into_unchecked_ref`
    pub fn into_unchecked_ref(self) -> &'a T {
        self.ptr
    }
}

impl<'a, T: 'a> Deref for Ref<'a, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self.ptr
    }
}

impl<'a, T: 'a> Drop for Ref<'a, T> {
    #[inline]
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        self.inner.release_imm_ref();
    }
}

/// The `Debug` implementation of `Ref` will just forward to the contained type.
impl<'a, T: fmt::Debug + 'a> fmt::Debug for Ref<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.ptr.fmt(f)
    }
}

/// A mutable reference to a `T`.
///
/// This is basically the same as a `&'a mut T`, but can do additional safety
/// checks when debug assertions are enabled.
///
/// It dereferences to a `&mut T`, but with a lifetime bound to `self`, not to
/// `'a`. This is because if the reference returned by `Deref`/`DerefMut` would
/// have lifetime `'a`, it could outlive the `RefMut`, which would make its
/// debug mode safety checks much less precise.
///
/// Since obtaining a `&'a mut T` is still safe when you uphold all the
/// invariants, it's still supported using the `into_unchecked_ref` method, but
/// requires the explicit method call.
///
/// # Safety
///
/// As long as an instance of `RefMut` exists, no other references to the same
/// value may be created (be it `Ref`s or `RefMut`s). This is the same rule that
/// Rust's native references follow.
pub struct RefMut<'a, T: 'a> {
    ptr: &'a mut T,
    #[cfg(debug_assertions)]
    inner: &'a Inner<T>,
}

impl<'a, T: 'a> RefMut<'a, T> {
    unsafe fn new(ptr: &'a Inner<T>) -> Self {
        ptr.obtain_mut_ref();

        // Create the mutable reference - since our reference and borrow
        // counting ensures that
        // 1. The value is still live
        // 2. We have exclusive access (via `obtain_mut_ref` methods)
        // this is safe to do in debug mode, provided no unchecked refs are
        // around.
        let value = &mut *ptr.value.get();

        Self {
            ptr: value,
            #[cfg(debug_assertions)]
            inner: ptr,
        }
    }

    /// Convert this `RefMut` into a normal Rust reference with lifetime `'a`.
    ///
    /// This will **disable** the debug mode borrow conflict check for this
    /// reference, but may be required when returning a mutable reference to
    /// some outside code. It will also **disable** the dangling pointer check
    /// for the returned reference, which means that you do not get a panic when
    /// you destroy all `UnsafeRef`s and then the `UnsafeBox` while there's
    /// still an unchecked Rust reference around.
    ///
    /// # Safety
    ///
    /// This function is generally safe to call if the invariants of the
    /// `RefMut` are upheld, which you have to do anyways, so this does not have
    /// to be marked as an unsafe function.
    // These docs are similar to the ones at `Ref::into_unchecked_ref`
    pub fn into_unchecked_ref(self) -> &'a mut T {
        self.ptr
    }
}

impl<'a, T: 'a> Deref for RefMut<'a, T> {
    #[inline]
    type Target = T;

    fn deref(&self) -> &T {
        self.ptr
    }
}

// Currently, the `Deref` impls do not return references of lifetime `'a`.
// This ensures that, by default, `RefMut` can properly track its lifetime and
// decrement the borrow count in its destructor.
impl<'a, T: 'a> DerefMut for RefMut<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        self.ptr
    }
}

impl<'a, T: 'a> Drop for RefMut<'a, T> {
    #[inline]
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        self.inner.release_mut_ref();
    }
}

/// The `Debug` implementation of `RefMut` will just forward to the contained
/// type.
impl<'a, T: fmt::Debug + 'a> fmt::Debug for RefMut<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.ptr.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicUsize, Ordering};

    // All of the `should_panic` tests do something the docs claim is UB, but
    // they are only enabled when we have debug asserts, which prevents the UB
    // due to our extra checks.

    #[test]
    fn drops_value() {
        static COUNT: AtomicUsize = AtomicUsize::new(0);

        struct NoisyDrop;

        impl Drop for NoisyDrop {
            fn drop(&mut self) {
                COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        assert_eq!(COUNT.load(Ordering::SeqCst), 0);
        let b = UnsafeBox::new(NoisyDrop);
        assert_eq!(COUNT.load(Ordering::SeqCst), 0);
        drop(b);
        assert_eq!(COUNT.load(Ordering::SeqCst), 1);
    }

    #[test]
    #[should_panic(expected = "there are still 1 UnsafeRefs around")]
    #[cfg_attr(not(debug_assertions), ignore)]   // test needs debug asserts
    fn outstanding_ref() {
        let mut b = UnsafeBox::new("some text".to_string());
        let _r = b.create_ref();
        drop(b);
    }

    #[test]
    #[should_panic(expected = "there are still 1000 UnsafeRefs around")]
    #[cfg_attr(not(debug_assertions), ignore)]   // test needs debug asserts
    fn many_outstanding_refs() {
        let mut b = UnsafeBox::new("some text".to_string());
        let mut refs = Vec::new();
        for _ in 0..1000 {
            refs.push(b.create_ref());
        }
        drop(b);
    }

    #[test]
    fn dropped_ref() {
        let mut b = UnsafeBox::new("some text".to_string());
        let r = b.create_ref();
        drop(r);
        drop(b);
    }

    #[test]
    fn many_dropped_refs() {
        let mut b = UnsafeBox::new("some text".to_string());
        let mut refs = Vec::new();
        for _ in 0..1000 {
            refs.push(b.create_ref());
        }
        drop(refs);
        drop(b);
    }

    #[test]
    fn unsafe_deref() {
        let mut b = UnsafeBox::new(vec![0, 1, 2]);
        let mut r = b.create_ref();

        {
            // Obtaining a reference needs an `unsafe` block
            let safe_ref = unsafe { r.unsafe_deref() };
            assert_eq!(*safe_ref, [0, 1, 2]);

            // UnsafeBox and UnsafeRef work the same way in this respect
            let safe_ref = unsafe { b.unsafe_deref() };
            assert_eq!(*safe_ref, [0, 1, 2]);

            // 2 immut. refs at the same time = fine
        }

        {
            // Try out mutating the value
            let mut mut_ref = unsafe { r.unsafe_deref_mut() };
            assert_eq!(*mut_ref, [0, 1, 2]);
            mut_ref[0] = -1;
            assert_eq!(*mut_ref, [-1, 1, 2]);
        }

        // Now getting an immutable ref should still work
        let safe_ref = unsafe { r.unsafe_deref() };
        assert_eq!(*safe_ref, [-1, 1, 2]);
    }

    #[test]
    #[should_panic(expected = "cannot borrow value mutably")]
    #[cfg_attr(not(debug_assertions), ignore)]   // test needs debug asserts
    fn double_mut_borrow() {
        let mut b = UnsafeBox::new("some text".to_string());
        let mut r = b.create_ref();
        let _m = unsafe { b.unsafe_deref_mut() };
        let _m2 = unsafe { r.unsafe_deref_mut() };
    }

    #[test]
    #[should_panic(expected = "cannot borrow value immutably")]
    #[cfg_attr(not(debug_assertions), ignore)]   // test needs debug asserts
    fn mut_immut_borrow() {
        let mut b = UnsafeBox::new("some text".to_string());
        let r = b.create_ref();
        let _m = unsafe { b.unsafe_deref_mut() };
        let _m2 = unsafe { r.unsafe_deref() };
    }

    #[test]
    #[should_panic(expected = "cannot borrow value mutably")]
    #[cfg_attr(not(debug_assertions), ignore)]   // test needs debug asserts
    fn immut_mut_borrow() {
        let mut b = UnsafeBox::new("some text".to_string());
        let r = b.create_ref();
        let _m = unsafe { r.unsafe_deref() };
        let _m2 = unsafe { b.unsafe_deref_mut() };
    }

    #[test]
    fn unchecked_refs() {
        // Ensure that `RefMut` and `Ref` resets the borrow count when consumed
        // via `into_unchecked_ref`.
        let mut b = UnsafeBox::new("some text".to_string());
        {
            let _unchecked_ref = unsafe { b.unsafe_deref() }.into_unchecked_ref();
        }
        {
            let _unchecked_ref = unsafe { b.unsafe_deref_mut() }.into_unchecked_ref();
        }
        {
            let _unchecked_ref = unsafe { b.unsafe_deref_mut() }.into_unchecked_ref();
        }
    }

    #[test]
    fn zsts() {
        let mut b = UnsafeBox::new(());
        unsafe {
            *b.unsafe_deref_mut() = ();
            let () = *b.unsafe_deref();
        }
        let mut r = b.create_ref();
        unsafe {
            *r.unsafe_deref_mut() = ();
            let () = *r.unsafe_deref();
        }
    }

    #[test]
    #[should_panic(expected = "there are still 1 UnsafeRefs around")]
    #[cfg_attr(not(debug_assertions), ignore)]   // test needs debug asserts
    fn zst_outstanding_ref() {
        let mut b = UnsafeBox::new(());
        let _r = b.create_ref();
        drop(b);
    }

    #[test]
    #[should_panic(expected = "cannot borrow value mutably")]
    #[cfg_attr(not(debug_assertions), ignore)]   // test needs debug asserts
    fn zst_double_mut_borrow() {
        let mut b = UnsafeBox::new(());
        let mut r = b.create_ref();
        let _m = unsafe { b.unsafe_deref_mut() };
        let _m2 = unsafe { r.unsafe_deref_mut() };
    }

    #[test]
    #[should_panic(expected = "cannot borrow value immutably")]
    #[cfg_attr(not(debug_assertions), ignore)]   // test needs debug asserts
    fn zst_mut_immut_borrow() {
        let mut b = UnsafeBox::new(());
        let r = b.create_ref();
        let _m = unsafe { b.unsafe_deref_mut() };
        let _m2 = unsafe { r.unsafe_deref() };
    }

    #[test]
    fn debug() {
        // Debug impls of `UnsafeBox`/`UnsafeRef` print the address of the value
        // Test that the output of both match
        let mut b = UnsafeBox::new(());
        let r = b.create_ref();

        assert_eq!(format!("{:?}", b).replace("UnsafeBox", "UnsafeRef"), format!("{:?}", r));

        let mut b = UnsafeBox::new(vec![0, 1, 2]);
        let r = b.create_ref();

        assert_eq!(format!("{:?}", b).replace("UnsafeBox", "UnsafeRef"), format!("{:?}", r));
    }
}
