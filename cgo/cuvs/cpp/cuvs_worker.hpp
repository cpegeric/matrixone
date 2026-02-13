#pragma once

#include <any>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
#include <iostream> // For temporary logging, should be replaced with a proper logging solution
#include <csignal>  // For signal handling
#include <cuda_runtime.h> // For cudaStreamCreate/Destroy

// For pinning threads to cores on Linux, similar to Go's LockOSThread
#ifdef __linux__
#include <pthread.h>
#endif

#include <raft/core/resources.hpp> // For raft::resources
#include <raft/core/resource/cuda_stream.hpp> // For raft::cuda_stream
#include <raft/core/handle.hpp> // For raft::handle (often embedded in resources)

// Define handle_t directly in the global namespace or in matrix_origin
// to avoid conflicts with cuvs's internal namespace resolution of raft types.
class RaftHandleWrapper {
public:
    // A raft::resources object manages CUDA streams, handles, and other components.
    std::unique_ptr<::raft::resources> resources_ = nullptr;

    RaftHandleWrapper();  // Constructor to create a raft::resources
    ~RaftHandleWrapper(); // Destructor to destroy the raft::resources

    // Getter for the underlying raft::resources object
    ::raft::resources* get_raft_resources() const { return resources_.get(); }
};

// Implementations for RaftHandleWrapper
inline RaftHandleWrapper::RaftHandleWrapper() {
    // raft::resources constructor often takes an existing stream or creates one.
    // Assuming default constructor creates an internal stream.
    resources_ = std::make_unique<::raft::resources>();
    // std::cout << "DEBUG: RAFT handle created with real raft::resources, stream " << resources_->get_cuda_stream() << std::endl;
}

inline RaftHandleWrapper::~RaftHandleWrapper() {
    if (resources_) {
        // raft::resources destructor handles cleanup of its internal stream and other components.
        resources_.reset();
    }
    // std::cout << "DEBUG: RAFT handle destroyed." << std::endl;
}

namespace matrix_origin {

// --- Forward Declarations for CuvsWorker related types ---
struct CuvsTaskResult;
class CuvsTaskResultStore;
class CuvsWorker;

// --- ThreadSafeQueue ---
/**
 * @brief A thread-safe, blocking queue.
 */
template <typename T>
class ThreadSafeQueue {
public:
    inline void push(T value) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push_back(std::move(value));
        }
        cond_.notify_one();
    }

    inline bool pop(T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [this] { return !queue_.empty() || stopped_; });
        if (stopped_ && queue_.empty()) {
            return false;
        }
        value = std::move(queue_.front());
        queue_.pop_front();
        return true;
    }

    inline void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        cond_.notify_all();
    }

    inline bool is_stopped() const { return stopped_; } // Added for checking stop status


private:
    std::deque<T> queue_;
    mutable std::mutex mutex_; // mutable for is_empty and is_stopped
    std::condition_variable cond_;
    bool stopped_ = false;
};

// --- CuvsTaskResult ---
/**
 * @brief Represents the result of a CuvsTask execution. Mirrors Go's CuvsTaskResult.
 */
struct CuvsTaskResult {
    uint64_t ID;
    std::any Result;
    std::exception_ptr Error;
};

// --- TaskState ---
/**
 * @brief Internal state for a task managed by CuvsTaskResultStore. Mirrors Go's taskState.
 */
struct TaskState {
    std::shared_ptr<std::promise<CuvsTaskResult>> promise_holder; // To signal completion
    std::shared_ptr<CuvsTaskResult> result_holder;              // To store the result once ready
    std::mutex mu;                                              // Protects access to result_holder and done
    std::condition_variable cv;                                 // For threads waiting for result
    bool done = false;                                          // True if result is available
};

// --- CuvsTaskResultStore ---
/**
 * @brief Manages the storage and retrieval of CuvsTaskResults. Mirrors Go's CuvsTaskResultStore.
 */
class CuvsTaskResultStore {
public:
    CuvsTaskResultStore();
    ~CuvsTaskResultStore();

    // Stores a result and signals any waiting threads.
    void Store(const CuvsTaskResult& result);

    // Waits until the result for the given jobID is available and returns a future to it.
    // Handles cases where Wait is called before or after Store.
    std::future<CuvsTaskResult> Wait(uint64_t jobID);

    // Atomically increments and returns a new unique job ID.
    uint64_t GetNextJobID();

    // Signals the store to stop, unblocking any waiting `Wait` calls.
    void Stop();

private:
    std::map<uint64_t, std::shared_ptr<TaskState>> states_;
    std::mutex mu_; // Protects states_ map
    std::atomic<uint64_t> next_job_id_;
    ThreadSafeQueue<bool> stop_channel_; // Simulates Go's stopCh
    std::atomic<bool> stopped_flag_;     // Simulates Go's atomic.Bool
};

// --- CuvsWorker ---
/**
 * @brief CuvsWorker runs tasks in a dedicated OS thread with a CUDA context.
 * Mirrors Go's CuvsWorker functionality closely.
 */
class CuvsWorker {
public:
    // Changed to use the globally defined RaftHandleWrapper
    using RaftHandle = RaftHandleWrapper; 
    // User-provided function type: takes a RaftHandle& and returns std::any, or throws.
    using UserTaskFn = std::function<std::any(RaftHandle&)>;

    // Internal representation of a task submitted to the worker.
    struct CuvsTask {
        uint64_t ID;
        UserTaskFn Fn;
    };

    /**
     * @brief Constructs a CuvsWorker.
     * @param n_threads The number of worker threads to use for task execution.
     */
    explicit CuvsWorker(size_t n_threads);

    /**
     * @brief Destructor. Calls stop() to ensure all threads are properly shut down.
     */
    ~CuvsWorker();

    // Deleted copy/move constructors and assignments to prevent accidental copying
    CuvsWorker(const CuvsWorker&) = delete;
    CuvsWorker& operator=(const CuvsWorker&) = delete;
    CuvsWorker(CuvsWorker&&) = delete;
    CuvsWorker& operator=(CuvsWorker&&) = delete;

    /**
     * @brief Starts the worker's execution loop.
     * @param init_fn An optional function to run once per resource initialization.
     * @param stop_fn An optional function to run once per resource deinitialization.
     */
    void Start(UserTaskFn init_fn = nullptr, UserTaskFn stop_fn = nullptr);

    /**
     * @brief Signals the worker to terminate and waits for all threads to finish.
     */
    void Stop();

    /**
     * @brief Submits a task for asynchronous execution.
     * @param fn The task function to execute.
     * @return A unique job ID for the submitted task.
     * @throws std::runtime_error if the worker is stopped.
     */
    uint64_t Submit(UserTaskFn fn);

    /**
     * @brief Blocks until the result for the given jobID is available and returns a future to it.
     * @param jobID The ID of the task to wait for.
     * @return A std::future<CuvsTaskResult> that will eventually hold the result.
     */
    std::future<CuvsTaskResult> Wait(uint64_t jobID);

    /**
     * @brief Returns the first internal error encountered by the worker.
     * @return An std::exception_ptr if an error occurred, otherwise nullptr.
     */
    std::exception_ptr GetFirstError();

private:
    // Helper function to set up a RaftHandleWrapper resource.
    std::unique_ptr<RaftHandle> setup_resource();

    // Processes a single CuvsTask and stores its result in the CuvsTaskResultStore.
    void handle_and_store_task(CuvsTask task, RaftHandle& resource);

    // Drains the tasks queue and processes remaining tasks during shutdown.
    void drain_and_process_tasks(RaftHandle& resource);

    // The main loop for the CuvsWorker, similar to Go's `run()` goroutine.
    void run_main_loop(UserTaskFn init_fn, UserTaskFn stop_fn);

    // The loop for individual worker threads, similar to Go's `workerLoop()` goroutines.
    void worker_sub_loop(std::shared_ptr<std::promise<void>> worker_ready_promise);

    // A separate thread for handling system signals (SIGTERM, SIGINT).
    void signal_handler_loop();

    size_t n_threads_;
    ThreadSafeQueue<CuvsTask> tasks_;         // Main task channel (Go's `tasks`)
    ThreadSafeQueue<bool> stop_channel_;      // For signaling stop (Go's `stopCh`)
    ThreadSafeQueue<std::exception_ptr> err_channel_; // For internal errors (Go's `errch`)

    std::thread main_run_thread_;            // Thread for run_main_loop
    std::thread signal_thread_;              // Thread for signal_handler_loop
    std::vector<std::thread> sub_workers_;   // Threads for worker_sub_loop

    std::atomic<bool> stopped_flag_{false};  // Worker's stopped status (Go's `stopped atomic.Bool`)
    std::atomic<bool> started_flag_{false};  // To prevent multiple starts

    CuvsTaskResultStore result_store_;       // Embedded result store

    std::mutex first_error_mu_;             // Mutex for first_error_
    std::exception_ptr first_error_;        // Stores the first encountered error
};

// --- Implementations for CuvsTaskResultStore ---

inline CuvsTaskResultStore::CuvsTaskResultStore() : next_job_id_(0), stopped_flag_(false) {}

inline CuvsTaskResultStore::~CuvsTaskResultStore() {
    Stop();
}

inline void CuvsTaskResultStore::Store(const CuvsTaskResult& result) {
    std::unique_lock<std::mutex> lock(mu_);
    auto it = states_.find(result.ID);
    if (it == states_.end()) {
        // This can happen if Wait() has not been called yet for this ID.
        // Create state and store result.
        auto state = std::make_shared<TaskState>();
        state->result_holder = std::make_shared<CuvsTaskResult>(result);
        state->done = true;
        states_[result.ID] = state;
        lock.unlock(); // Release map lock before notifying
        state->cv.notify_all();
    } else {
        // Wait() was called, state already exists.
        auto state = it->second;
        std::lock_guard<std::mutex> state_lock(state->mu);
        state->result_holder = std::make_shared<CuvsTaskResult>(result);
        state->done = true;
        lock.unlock(); // Release map lock before notifying
        state->cv.notify_all();
    }
}

inline std::future<CuvsTaskResult> CuvsTaskResultStore::Wait(uint64_t jobID) {
    std::shared_ptr<TaskState> state;
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = states_.find(jobID);
        if (it == states_.end()) {
            // Task not submitted/stored yet, create state and wait.
            state = std::make_shared<TaskState>();
            states_[jobID] = state;
        } else {
            // Task already in map, use existing state.
            state = it->second;
        }
    }

    // Now, outside the map lock, wait on the task-specific condition variable.
    // If a promise exists, associate the future with it.
    if (!state->promise_holder) {
        state->promise_holder = std::make_shared<std::promise<CuvsTaskResult>>();
    }

    // Wait for the result to be ready
    std::unique_lock<std::mutex> state_lock(state->mu);
    state->cv.wait(state_lock, [&]() {
        return state->done || stopped_flag_.load();
    });

    if (stopped_flag_.load()) {
        // If store stopped while waiting, set an exception for the future.
        state->promise_holder->set_exception(
            std::make_exception_ptr(std::runtime_error("CuvsTaskResultStore stopped before result was available"))
        );
        std::lock_guard<std::mutex> lock(mu_);
        states_.erase(jobID); // Clean up state
        return state->promise_holder->get_future();
    }

    // Result is available, fulfill the promise.
    if (state->result_holder) {
        state->promise_holder->set_value(*state->result_holder);
    } else {
        // This case should ideally not happen if state->done is true and no error occurred.
        state->promise_holder->set_exception(
            std::make_exception_ptr(std::runtime_error("CuvsTaskResultStore: Result holder was null after done signal"))
        );
    }
    
    // Remove after retrieval, similar to Go.
    std::lock_guard<std::mutex> lock(mu_);
    states_.erase(jobID);
    return state->promise_holder->get_future();
}


inline uint64_t CuvsTaskResultStore::GetNextJobID() {
    return next_job_id_.fetch_add(1) + 1; // Increment and return, matching Go's 1-based start.
}

inline void CuvsTaskResultStore::Stop() {
    bool expected = false;
    if (stopped_flag_.compare_exchange_strong(expected, true)) {
        stop_channel_.push(true); // Signal stop, unblock any ongoing waits
        // Notify all waiting condition variables in states_ map
        std::lock_guard<std::mutex> lock(mu_);
        for (auto const& [id, state] : states_) {
            state->cv.notify_all();
        }
    }
}


// --- Implementations for CuvsWorker ---

// Static signal handler, needs to forward to an instance if used in a class.
// For simplicity, we directly handle signals in a dedicated thread.
inline static std::atomic<bool> global_signal_received(false);
inline static void signal_handler(int signum) {
    std::cout << "DEBUG: Signal " << signum << " received." << std::endl;
    global_signal_received.store(true);
}


inline CuvsWorker::CuvsWorker(size_t n_threads) : n_threads_(n_threads) {
    if (n_threads_ == 0) {
        throw std::invalid_argument("CuvsWorker thread count must be non-zero.");
    }
}

inline CuvsWorker::~CuvsWorker() {
    Stop();
}

inline std::unique_ptr<CuvsWorker::RaftHandle> CuvsWorker::setup_resource() {
    try {
        auto res = std::make_unique<RaftHandle>();
        return res;
    } catch (const std::exception& e) {
        err_channel_.push(std::current_exception());
        std::cerr << "ERROR: Failed to setup RAFT resource: " << e.what() << std::endl;
        return nullptr;
    }
}

inline void CuvsWorker::handle_and_store_task(CuvsTask task, RaftHandle& resource) {
    CuvsTaskResult cuvs_result;
    cuvs_result.ID = task.ID;
    try {
        cuvs_result.Result = task.Fn(resource);
    } catch (const std::exception& e) {
        cuvs_result.Error = std::current_exception();
        // Log the error
        std::cerr << "ERROR: Task " << task.ID << " failed: " << e.what() << std::endl;
    } catch (...) {
        cuvs_result.Error = std::current_exception();
        // Log unknown error
        std::cerr << "ERROR: Task " << task.ID << " failed with unknown exception." << std::endl;
    }
    result_store_.Store(cuvs_result);
}

inline void CuvsWorker::drain_and_process_tasks(RaftHandle& resource) {
    CuvsTask task;
    while (tasks_.pop(task)) {
        handle_and_store_task(task, resource);
    }
}

inline void CuvsWorker::worker_sub_loop(std::shared_ptr<std::promise<void>> worker_ready_promise) {
#ifdef __linux__
    static std::atomic<int> cpu_idx = 0;
    if (std::thread::hardware_concurrency() > 0) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        int core_id = cpu_idx.fetch_add(1) % std::thread::hardware_concurrency();
        CPU_SET(core_id, &cpuset);
        if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
            std::cerr << "WARNING: Failed to set affinity for worker thread to core " << core_id << std::endl;
        }
    }
#endif

    auto resource = setup_resource();
    if (!resource) {
        worker_ready_promise->set_exception(
            std::make_exception_ptr(std::runtime_error("Worker failed to setup resource."))
        );
        return;
    }
    // Signal that this worker is ready
    worker_ready_promise->set_value();

    while (true) {
        CuvsTask task;
        if (!tasks_.pop(task)) {
            // Queue is stopped and empty, or global_stop_flag_ is set
            break;
        }
        handle_and_store_task(task, *resource);
    }
    // Drain any remaining tasks if stop was called, but tasks were still in queue
    drain_and_process_tasks(*resource);
}

inline void CuvsWorker::run_main_loop(UserTaskFn init_fn, UserTaskFn stop_fn) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset); // Pin main loop to core 0, or some other designated core
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        std::cerr << "WARNING: Failed to set affinity for main_run_loop to core 0" << std::endl;
    }
#endif

    auto parent_resource = setup_resource();
    if (!parent_resource) {
        std::cerr << "FATAL: Main loop failed to setup parent resource." << std::endl;
        // The error is already pushed to err_channel_ by setup_resource
        return;
    }

    if (init_fn) {
        try {
            init_fn(*parent_resource);
        } catch (const std::exception& e) {
            std::exception_ptr current_ex = std::current_exception();
            err_channel_.push(current_ex);
            // Also set first_error_ immediately if it's the first one
            if (!first_error_) {
                std::lock_guard<std::mutex> lock(first_error_mu_);
                if (!first_error_) { first_error_ = current_ex; }
            }
            std::cerr << "ERROR: initFn failed: " << e.what() << std::endl;
            stop_channel_.push(true); // Signal main loop to stop immediately
            return;
        }
    }

    // Ensure stopFn is called when exiting this scope
    auto stop_fn_defer = [&]() {
        if (stop_fn) {
            try {
                stop_fn(*parent_resource);
            } catch (const std::exception& e) {
                err_channel_.push(std::current_exception());
                std::cerr << "ERROR: stopFn failed: " << e.what() << std::endl;
            }
        }
    };
    // Use a lambda with a local variable to simulate defer
    std::shared_ptr<void> _(nullptr, [&](...) { stop_fn_defer(); });


    if (n_threads_ == 1) {
        // Special case: nthread is 1, process tasks directly in this thread
        while (!stop_channel_.is_stopped() && !err_channel_.is_stopped()) {
            CuvsTask task;
            if (tasks_.pop(task)) {
                handle_and_store_task(task, *parent_resource);
            }
        }
        // Drain any remaining tasks if stop was called
        drain_and_process_tasks(*parent_resource);
    } else {
        // General case: nthread > 1, create worker threads
        std::vector<std::shared_ptr<std::promise<void>>> worker_ready_promises(n_threads_);
        std::vector<std::future<void>> worker_ready_futures(n_threads_);

        sub_workers_.reserve(n_threads_);
        for (size_t i = 0; i < n_threads_; ++i) {
            worker_ready_promises[i] = std::make_shared<std::promise<void>>();
            worker_ready_futures[i] = worker_ready_promises[i]->get_future();
            sub_workers_.emplace_back(&CuvsWorker::worker_sub_loop, this, worker_ready_promises[i]);
        }

        // Wait for all sub-workers to be ready
        try {
            for (auto& f : worker_ready_futures) {
                f.get(); // Will rethrow exception if worker setup failed
            }
        } catch (const std::exception& e) {
            err_channel_.push(std::current_exception());
            std::cerr << "ERROR: One or more sub-workers failed to initialize: " << e.what() << std::endl;
            stop_channel_.push(true); // Signal main loop to stop
        }
        
        // Wait until stop is signaled or an error occurs
        bool dummy;
        std::exception_ptr err_ptr;
        while (!stop_channel_.is_stopped() && !err_channel_.is_stopped()) {
            if (stop_channel_.pop(dummy)) { break; } // stop signal received
            if (err_channel_.pop(err_ptr)) { // Error received from internal channel
                if (!first_error_) {
                    std::lock_guard<std::mutex> lock(first_error_mu_);
                    if (!first_error_) { first_error_ = err_ptr; }
                }
                stop_channel_.push(true); // Signal main loop to stop
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Prevent busy waiting
        }

        // Join all sub-workers
        for (auto& worker : sub_workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }
    std::cout << "DEBUG: CuvsWorker main loop finished." << std::endl;
}

inline void CuvsWorker::signal_handler_loop() {
    // This thread will effectively take over signal handling,
    // as signals are delivered to one arbitrary thread in the process.
    // For simplicity, we directly handle signals in a dedicated thread.
    // In a production system, you might use sigwaitinfo for specific signals.

    std::signal(SIGTERM, signal_handler);
    std::signal(SIGINT, signal_handler);

    std::cout << "DEBUG: Signal handler thread started." << std::endl;

    while (!stopped_flag_.load()) {
        if (global_signal_received.load()) {
            std::cout << "DEBUG: CuvsWorker received shutdown signal, stopping..." << std::endl;
            stop_channel_.push(true); // Signal main loop to stop
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    std::cout << "DEBUG: Signal handler thread finished." << std::endl;
}


inline void CuvsWorker::Start(UserTaskFn init_fn, UserTaskFn stop_fn) {
    bool expected = false;
    if (!started_flag_.compare_exchange_strong(expected, true)) {
        std::cerr << "WARNING: CuvsWorker already started." << std::endl;
        return;
    }

    main_run_thread_ = std::thread(&CuvsWorker::run_main_loop, this, init_fn, stop_fn);
    signal_thread_ = std::thread(&CuvsWorker::signal_handler_loop, this);
}

inline void CuvsWorker::Stop() {
    bool expected = false;
    if (stopped_flag_.compare_exchange_strong(expected, true)) {
        std::cout << "DEBUG: CuvsWorker Stop() called." << std::endl;
        // Signal all internal queues/channels to stop
        stop_channel_.push(true); // Signal main_run_loop to stop
        tasks_.stop();            // Stop task queue
        err_channel_.stop();      // Stop error channel
        result_store_.Stop();     // Stop result store

        // Join all worker threads
        if (main_run_thread_.joinable()) {
            main_run_thread_.join();
        }
        if (signal_thread_.joinable()) {
            signal_thread_.join();
        }
        for (auto& worker : sub_workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        sub_workers_.clear();
        started_flag_.store(false); // Allow restarting if desired
        std::cout << "DEBUG: CuvsWorker Stop() completed." << std::endl;
    }
}

inline uint64_t CuvsWorker::Submit(UserTaskFn fn) {
    if (stopped_flag_.load()) {
        throw std::runtime_error("cannot submit task: worker is stopped");
    }
    uint64_t jobID = result_store_.GetNextJobID();
    CuvsTask task = {jobID, std::move(fn)};
    tasks_.push(std::move(task));
    return jobID;
}

inline std::future<CuvsTaskResult> CuvsWorker::Wait(uint64_t jobID) {
    return result_store_.Wait(jobID);
}

inline std::exception_ptr CuvsWorker::GetFirstError() {
    std::lock_guard<std::mutex> lock(first_error_mu_);
    return first_error_;
}

} // namespace matrix_origin