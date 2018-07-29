//! A data structure that allows looking up items using one of two properties.
//!
//! This example aims to illustrate usage of the `unsafe_box` crate to store two
//! pointers to a heap-allocated value and build what is hopefully a safe
//! abstraction.

extern crate unsafe_box;

use unsafe_box::*;
use std::collections::btree_map::*;

struct Storage<T> {
    // Which `BTreeMap` gets the `UnsafeBox` doesn't matter too much, but you
    // must always destroy all `UnsafeRef`s *first*.
    // Also note that Rust will destroy structs starting with the first field,
    // so if we keep the refs in there we won't have to write a drop impl.

    by_id: BTreeMap<u32, UnsafeRef<Item<T>>>,
    by_name: BTreeMap<String, UnsafeBox<Item<T>>>,
}

struct Item<T> {
    name: String,
    id: u32,
    value: T,   // might be modified later on
}

impl<T> Storage<T> {
    fn new() -> Self {
        Storage {
            by_id: BTreeMap::new(),
            by_name: BTreeMap::new(),
        }
    }

    fn insert(&mut self, name: String, id: u32, value: T) {
        // Create the item and move it to its final heap allocation and grab an
        // `UnsafeRef` pointing to it.
        let mut b = UnsafeBox::new(Item {
            name: name.clone(),
            id,
            value,
        });
        let r = b.create_ref();

        // Put box and ref in their place. To avoid issues where a value is
        // added that overwrites only the id or name of an existing value, we
        // use the entry API and prevent this.
        // If we wouldn't, this could easily cause memory problems due to
        // dangling pointers when overwriting an existing entry in only one map.
        let (id_entry, name_entry) = (self.by_id.entry(id), self.by_name.entry(name));
        match (id_entry, name_entry) {
            (Entry::Vacant(id_vacant), Entry::Vacant(name_vacant)) => {
                // Both entries are free, so there's no problem
                id_vacant.insert(r);
                name_vacant.insert(b);
            }
            _ => {
                // At least one entry is already occupied - Invalid operation.
                panic!("attempted to overwrite existing value - `remove` overlapping items first");
            }
        }
    }

    fn remove_by_name(&mut self, name: String) {
        if let Entry::Occupied(entry) = self.by_name.entry(name) {
            let item = entry.remove();
            // Safe: We have mutable access to `Storage` and don't keep any
            // other refs around.
            let id = unsafe { item.unsafe_deref().id };
            self.by_id.remove(&id);  // destroy the `UnsafeRef`
        }
    }

    fn remove_by_id(&mut self, id: u32) {
        if let Entry::Occupied(entry) = self.by_id.entry(id) {
            let item = entry.remove();
            let unsafe_box = {
                // Safe: We have mutable access to `Storage` and don't keep any
                // other refs around.
                let imm_ref = unsafe { item.unsafe_deref() };
                self.by_name.remove(&imm_ref.name)
            };
            // Destroy the `UnsafeRef` first
            drop(item);
            // Now we can destroy the `UnsafeBox` we removed from the entry.
            drop(unsafe_box);
        }
    }

    fn get_by_name<'a>(&'a mut self, name: &str) -> Option<&'a mut T> {
        // Using `get` would work too, since `UnsafeBox` effectively allows
        // interior mutability. The `get_mut` just makes clear what's happening.
        if let Some(unsafe_box) = self.by_name.get_mut(name) {
            // Safe: We have mutable access to `self`, and the return values
            // lifetime is bound to that mutable `self` reference. This prevents
            // any usage of `self` while that reference is around.
            let ref_mut: RefMut<_> = unsafe { unsafe_box.unsafe_deref_mut() };

            // Now we have to call `into_unchecked_ref` to convert the `RefMut`
            // into a normal Rust reference. This will disable the extra safety
            // checks in debug mode, so be extra careful when doing so.
            // While we *could* just reborrow the `RefMut`, the resulting
            // lifetime would be too short to return it (it cannot outlive the
            // `RefMut` it's created from).
            Some(&mut ref_mut.into_unchecked_ref().value)
        } else {
            None
        }
    }

    fn get_by_id<'a>(&'a mut self, id: u32) -> Option<&'a mut T> {
        // This works just like `get_by_name`, except we get the reference to
        // the value from an `UnsafeRef` instead of a box.
        if let Some(unsafe_ref) = self.by_id.get_mut(&id) {
            let ref_mut: RefMut<_> = unsafe { unsafe_ref.unsafe_deref_mut() };
            Some(&mut ref_mut.into_unchecked_ref().value)
        } else {
            None
        }
    }
}

fn main() {
    let mut storage = Storage::new();

    storage.insert("test".to_string(), 123, vec![0, 1, 2, 3]);
    storage.insert("blabla".to_string(), 0, vec![-987]);
    storage.insert("asdasdasdasd".to_string(), 1, vec![0]);
    assert_eq!(storage.get_by_name("test").unwrap(), &[0, 1, 2, 3]);
    assert_eq!(storage.get_by_id(123).unwrap(), &[0, 1, 2, 3]);

    // The following will panic since "test" is already used as a name
    //storage.insert("test".to_string(), 2, vec![]);

    storage.remove_by_name("test".to_string());
    assert!(storage.get_by_name("test").is_none());
    assert!(storage.get_by_id(123).is_none());

    // Now that the colliding item is removed, we can try again
    storage.insert("test".to_string(), 2, vec![]);

    storage.remove_by_id(0);
    assert!(storage.get_by_id(0).is_none());
    assert!(storage.get_by_name("blabla").is_none());
}
