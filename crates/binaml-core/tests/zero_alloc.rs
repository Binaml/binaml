use binaml_core::{
    BClassifier, BRegressor, FunctionBuildConfig, FunctionBuildSession, SignBatch, DEFAULT_L_PAT,
    DEFAULT_MAX_EXPERT_NODES,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static ALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATION_COUNT.fetch_add(1, Ordering::SeqCst);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn reset_alloc_counter() {
    ALLOCATION_COUNT.store(0, Ordering::SeqCst);
}

fn alloc_count() -> usize {
    ALLOCATION_COUNT.load(Ordering::SeqCst)
}

#[test]
fn zero_alloc_after_new_on_ensemble_hot_paths() {
    {
        let mut model =
            BRegressor::with_hyperparameters(4, 0.1, 0.0, 4, 1, 4, 8, 32, 2).expect("valid config");
        reset_alloc_counter();

        for step in 0..12 {
            let features = [step % 2 == 0, step % 3 == 0, step % 5 == 0, step % 7 == 0];
            let target = if step % 2 == 0 { 1.0 } else { -1.0 };
            model.predict(&features).expect("predict");
            model.update(target).expect("update");
        }

        assert_eq!(
            alloc_count(),
            0,
            "regressor ensemble hot path allocated after construction"
        );
    }

    {
        let mut model = BClassifier::with_hyperparameters(2, 3, 0.1, 0.0, 1, 1, 8, 8, 64, 2)
            .expect("valid config");
        reset_alloc_counter();

        for step in 0..12 {
            let features = if step % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            };
            let target = step % 3;
            model.predict(&features).expect("predict");
            model.update(target).expect("update");
        }

        assert_eq!(
            alloc_count(),
            0,
            "classifier ensemble hot path allocated after construction"
        );
    }

    {
        let config = FunctionBuildConfig::new(4, 2, 2, DEFAULT_MAX_EXPERT_NODES, DEFAULT_L_PAT);
        let mut session = FunctionBuildSession::new(config, 2).expect("valid config");
        reset_alloc_counter();
        let first = [false, true, false, true];
        let second = [false, false, true, true];
        let signs = [false, true, true, true];
        for _ in 0..8 {
            let columns = [&first[..], &second[..]];
            session
                .build(SignBatch::from_columns(&columns, &signs))
                .expect("build");
        }
        assert_eq!(
            alloc_count(),
            0,
            "function build session allocated after construction"
        );
    }
}
