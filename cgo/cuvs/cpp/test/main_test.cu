#include "cuvs_worker.hpp" // Include your main code
#include "test_framework.hpp"

// Define the thread_local variable declared in test_framework.hpp
thread_local bool current_test_failed = false;


// Forward declare the namespace for convenience
using namespace matrixone;

// Helper to check if an exception_ptr holds a specific exception type
// template <typename E>
// bool has_exception(const std::exception_ptr& ep) {
//     if (!ep) return false;
//     try {
//         std::rethrow_exception(ep);
//     } catch (const E& e) {
//         return true;
//     } catch (...) {
//         return false;
//     }
// }

// --- ThreadSafeQueue Tests ---

TEST(ThreadSafeQueueTest, PushAndPop) {
    ThreadSafeQueue<int> queue;
    queue.push(1);
    int val;
    ASSERT_TRUE(queue.pop(val));
    ASSERT_EQ(val, 1);
}

TEST(ThreadSafeQueueTest, MultiplePushesAndPops) {
    ThreadSafeQueue<int> queue;
    queue.push(1);
    queue.push(2);
    queue.push(3);

    int val;
    ASSERT_TRUE(queue.pop(val));
    ASSERT_EQ(val, 1);
    ASSERT_TRUE(queue.pop(val));
    ASSERT_EQ(val, 2);
    ASSERT_TRUE(queue.pop(val));
    ASSERT_EQ(val, 3);
}

TEST(ThreadSafeQueueTest, PopBlocksWhenEmpty) {
    ThreadSafeQueue<int> queue;
    std::atomic<bool> popped(false);
    std::thread t([&]() {
        int val;
        queue.pop(val); // This should block
        popped.store(true);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Give thread time to block
    ASSERT_FALSE(popped.load());

    queue.push(42);
    t.join();
    ASSERT_TRUE(popped.load());
}

TEST(ThreadSafeQueueTest, StopUnblocksPop) {
    ThreadSafeQueue<int> queue;
    std::atomic<bool> pop_returned(false);
    std::thread t([&]() {
        int val;
        bool result = queue.pop(val); // Should return false if stopped and empty
        ASSERT_FALSE(result);         // Assert within the thread
        pop_returned.store(true);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Give thread time to block
    ASSERT_FALSE(pop_returned.load());

    queue.stop();
    t.join();
    ASSERT_TRUE(pop_returned.load());
}

TEST(ThreadSafeQueueTest, ConcurrentAccess) {
    ThreadSafeQueue<int> queue;
    const int num_threads = 5;
    const int num_items_per_thread = 1000;
    std::vector<std::thread> producers;
    std::vector<int> consumed_items;
    std::mutex consumed_mu;

    for (int i = 0; i < num_threads; ++i) {
        producers.emplace_back([&queue, i, num_items_per_thread]() {
            for (int j = 0; j < num_items_per_thread; ++j) {
                queue.push(i * num_items_per_thread + j);
            }
        });
    }

    std::thread consumer([&]() {
        for (int i = 0; i < num_threads * num_items_per_thread; ++i) {
            int val;
            ASSERT_TRUE(queue.pop(val));
            std::lock_guard<std::mutex> lock(consumed_mu);
            consumed_items.push_back(val);
        }
    });

    for (auto& t : producers) {
        t.join();
    }
    queue.stop(); // Consumer might still be running if it hasn't popped everything yet
    consumer.join();

    ASSERT_EQ(consumed_items.size(), (size_t)(num_threads * num_items_per_thread));
    std::sort(consumed_items.begin(), consumed_items.end());
    for (int i = 0; i < num_threads * num_items_per_thread; ++i) {
        ASSERT_EQ(consumed_items[i], i);
    }
}

// --- CuvsTaskResultStore Tests ---

TEST(CuvsTaskResultStoreTest, StoreThenWait) {
    CuvsTaskResultStore store;
    uint64_t jobID = store.GetNextJobID();
    CuvsTaskResult result{jobID, std::string("Success"), nullptr};

    store.Store(result);
    std::future<CuvsTaskResult> future = store.Wait(jobID);
    
    CuvsTaskResult retrieved_result = future.get();
    ASSERT_EQ(retrieved_result.Result.type().name(), typeid(std::string).name());
    ASSERT_EQ(std::any_cast<std::string>(retrieved_result.Result), "Success");
    ASSERT_FALSE(retrieved_result.Error);
}

TEST(CuvsTaskResultStoreTest, WaitThenStore) {
    CuvsTaskResultStore store;
    uint64_t jobID = store.GetNextJobID();

    std::future<CuvsTaskResult> future = std::async(std::launch::async, [&]() {
        return store.Wait(jobID).get();
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Give async thread time to call Wait
    
    CuvsTaskResult result{jobID, 123, nullptr};
    store.Store(result);

    CuvsTaskResult retrieved_result = future.get();
    ASSERT_EQ(retrieved_result.Result.type().name(), typeid(int).name());
    ASSERT_EQ(std::any_cast<int>(retrieved_result.Result), 123);
    ASSERT_FALSE(retrieved_result.Error);
}

TEST(CuvsTaskResultStoreTest, WaitWithError) {
    CuvsTaskResultStore store;
    uint64_t jobID = store.GetNextJobID();

    std::future<CuvsTaskResult> future = std::async(std::launch::async, [&]() {
        return store.Wait(jobID).get();
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    CuvsTaskResult result{jobID, std::any(), std::make_exception_ptr(std::runtime_error("Test Error"))};
    store.Store(result);

    CuvsTaskResult retrieved_result = future.get();
    ASSERT_TRUE(retrieved_result.Error);
    ASSERT_TRUE(has_exception<std::runtime_error>(retrieved_result.Error));
}

TEST(CuvsTaskResultStoreTest, StopUnblocksWait) {
    CuvsTaskResultStore store;
    uint64_t jobID = store.GetNextJobID();

    std::atomic<bool> wait_returned(false);
    std::thread t([&]() {
        try {
            store.Wait(jobID).get();
        } catch (const std::runtime_error& e) {
            ASSERT_EQ(std::string(e.what()), std::string("CuvsTaskResultStore stopped before result was available"));
        } catch (...) {
            ASSERT_TRUE(false); // Fail if unexpected exception type
        }
        wait_returned.store(true);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    ASSERT_FALSE(wait_returned.load());

    store.Stop();
    t.join();
    ASSERT_TRUE(wait_returned.load());
}

TEST(CuvsTaskResultStoreTest, GetNextJobIDIncrements) {
    CuvsTaskResultStore store;
    uint64_t id1 = store.GetNextJobID();
    uint64_t id2 = store.GetNextJobID();
    ASSERT_EQ(id2, id1 + 1);
}

// --- CuvsWorker Tests ---

// Simple task function for testing
std::any test_task_fn(RaftHandleWrapper& resource) {
    (void)resource; // Unused in this simple test
    // Simulate some work
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return std::string("TaskDone");
}

std::any test_task_fn_with_exception(RaftHandleWrapper& resource) {
    (void)resource;
    throw std::runtime_error("Task exception!");
}

std::any test_init_fn(RaftHandleWrapper& resource) {
    (void)resource;
    TEST_LOG("initFn called");
    return std::any(); // initFn does not return value in Go, so std::any() is fine.
}

std::any test_stop_fn(RaftHandleWrapper& resource) {
    (void)resource;
    TEST_LOG("stopFn called");
    return std::any();
}

TEST(CuvsWorkerTest, BasicTaskSubmissionAndWait) {
    CuvsWorker worker(1);
    worker.Start();

    uint64_t jobID = worker.Submit(test_task_fn);
    std::future<CuvsTaskResult> future = worker.Wait(jobID);
    
    CuvsTaskResult result = future.get();
    ASSERT_EQ(result.ID, jobID);
    ASSERT_EQ(result.Result.type().name(), typeid(std::string).name());
    ASSERT_EQ(std::any_cast<std::string>(result.Result), "TaskDone");
    ASSERT_FALSE(result.Error);

    worker.Stop();
}

TEST(CuvsWorkerTest, MultipleTasksWithMultipleThreads) {
    const size_t num_threads = 4;
    const size_t num_tasks = 20;
    CuvsWorker worker(num_threads);
    worker.Start();

    std::vector<uint64_t> job_ids;
    for (size_t i = 0; i < num_tasks; ++i) {
        job_ids.push_back(worker.Submit(test_task_fn));
    }

    for (uint64_t jobID : job_ids) {
        std::future<CuvsTaskResult> future = worker.Wait(jobID);
        CuvsTaskResult result = future.get();
        ASSERT_EQ(result.ID, jobID);
        ASSERT_EQ(result.Result.type().name(), typeid(std::string).name());
        ASSERT_EQ(std::any_cast<std::string>(result.Result), "TaskDone");
        ASSERT_FALSE(result.Error);
    }
    worker.Stop();
}

TEST(CuvsWorkerTest, TaskThrowsException) {
    CuvsWorker worker(1);
    worker.Start();

    uint64_t jobID = worker.Submit(test_task_fn_with_exception);
    std::future<CuvsTaskResult> future = worker.Wait(jobID);

    CuvsTaskResult result = future.get();
    ASSERT_EQ(result.ID, jobID);
    ASSERT_TRUE(result.Error);
    ASSERT_TRUE(has_exception<std::runtime_error>(result.Error));
    worker.Stop();
}

TEST(CuvsWorkerTest, InitAndStopFunctionsCalled) {
    // We'll use atomics to track if init/stop fns are called.
    std::atomic<bool> init_called(false);
    std::atomic<bool> stop_called(false);

    auto custom_init_fn = [&](RaftHandleWrapper& resource) -> std::any {
        init_called.store(true);
        return test_init_fn(resource);
    };

    auto custom_stop_fn = [&](RaftHandleWrapper& resource) -> std::any {
        stop_called.store(true);
        return test_stop_fn(resource);
    };

    CuvsWorker worker(1); // With n_threads=1, init/stop are called once on the parent resource
    worker.Start(custom_init_fn, custom_stop_fn);

    // Give some time for initFn to be called in the main loop
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ASSERT_TRUE(init_called.load());
    ASSERT_FALSE(stop_called.load()); // Stop should not be called yet

    worker.Stop();
    // After stopping, stopFn should have been called
    ASSERT_TRUE(stop_called.load());
}

TEST(CuvsWorkerTest, GetFirstError) {
    // Let's make initFn throw to test GetFirstError
    auto init_fn_that_throws = [](RaftHandleWrapper& resource) -> std::any {
        (void)resource;
        throw std::runtime_error("Init function failed intentionally");
    };

    CuvsWorker error_worker(1);
    error_worker.Start(init_fn_that_throws, nullptr);

    // Give time for the error to propagate
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::exception_ptr first_err = error_worker.GetFirstError();
    ASSERT_TRUE(first_err);
    ASSERT_TRUE(has_exception<std::runtime_error>(first_err));

    error_worker.Stop(); // Ensure clean shutdown
}

TEST(CuvsWorkerTest, SubmitToStoppedWorkerFails) {
    CuvsWorker worker(1);
    worker.Start();
    worker.Stop();

    ASSERT_THROW(worker.Submit(test_task_fn), std::runtime_error);
}

// Additional test case for n_threads > 1 to ensure sub-workers initialize correctly
TEST(CuvsWorkerTest, MultipleThreadsInitCorrectly) {
    const size_t num_threads = 4;
    CuvsWorker worker(num_threads);
    worker.Start();
    // Give some time for all worker_sub_loop to start and setup their resources.
    // If any setup_resource fails, it would push to err_channel_ and potentially stop the main loop.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Check if any internal errors were captured during startup of sub-workers
    ASSERT_FALSE(worker.GetFirstError());
    worker.Stop();
}

int main() {
    return RUN_ALL_TESTS();
}
