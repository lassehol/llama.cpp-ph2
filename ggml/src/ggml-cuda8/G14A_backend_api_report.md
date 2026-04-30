# G14A Backend API Inspection Report

Generated: `2026-04-29 17:11:37 UTC`

Repository root:

```text
/workspace/notebooks/llama.cpp-ph2
```

## Headers inspected

- `ggml/include/ggml-backend.h`
- `ggml/src/ggml-backend-impl.h`

## Structs found

### `ggml/src/ggml-backend-impl.h`

- `ggml_backend_i` lines 105-140
- `ggml_backend_buffer_i` lines 41-62
- `ggml_backend_buffer_type_i` lines 17-29
- `ggml_backend_device_i` lines 160-202
- `ggml_backend_reg_i` lines 214-224

## `ggml_backend_i` callback shape

Source: `ggml/src/ggml-backend-impl.h` lines 105-140

Detected `ggml_backend_i` callback fields:
- `get_name`
- `free`
- `set_tensor_async`
- `get_tensor_async`
- `set_tensor_2d_async`
- `get_tensor_2d_async`
- `cpy_tensor_async`
- `synchronize`
- `graph_plan_create`
- `graph_plan_free`
- `graph_plan_update`
- `graph_plan_compute`
- `graph_compute`
- `event_record`
- `event_wait`
- `graph_optimize`

Graph/compute-shaped fields detected: `graph_plan_create`, `graph_plan_free`, `graph_plan_update`, `graph_plan_compute`, `graph_compute`, `graph_optimize`
No buffer-shaped fields were detected in `ggml_backend_i` by name.
No device-shaped fields were detected in `ggml_backend_i` by name.

Raw struct:

```c
struct ggml_backend_i {
        const char * (*get_name)(ggml_backend_t backend);

        void (*free)(ggml_backend_t backend);

        // (optional) asynchronous tensor data access
        void (*set_tensor_async)   (ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void (*get_tensor_async)   (ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        void (*set_tensor_2d_async)(ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
        void (*get_tensor_2d_async)(ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
        bool (*cpy_tensor_async)(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);

        // (optional) complete all pending operations (required if the backend supports async operations)
        void (*synchronize)(ggml_backend_t backend);

        // (optional) graph plans (not used currently)
        // compute graph with a plan
        ggml_backend_graph_plan_t (*graph_plan_create) (ggml_backend_t backend, const struct ggml_cgraph * cgraph);
        void                      (*graph_plan_free)   (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
        // update the plan with a new graph - this should be faster than creating a new plan when the graph has the same topology
        void                      (*graph_plan_update) (ggml_backend_t backend, ggml_backend_graph_plan_t plan, const struct ggml_cgraph * cgraph);
        // compute the graph with the plan
        enum ggml_status          (*graph_plan_compute)(ggml_backend_t backend, ggml_backend_graph_plan_t plan);

        // compute graph (always async if supported by the backend)
        enum ggml_status          (*graph_compute)     (ggml_backend_t backend, struct ggml_cgraph * cgraph);

        // (optional) event synchronization
        // record an event on this stream
        void (*event_record)(ggml_backend_t backend, ggml_backend_event_t event);
        // wait for an event on on a different stream
        void (*event_wait)  (ggml_backend_t backend, ggml_backend_event_t event);

        // (optional) sort/optimize the nodes in the graph
        void                      (*graph_optimize)    (ggml_backend_t backend, struct ggml_cgraph * cgraph);
    };
```

## Keyword scan

Lines containing one of: `graph`, `compute`, `sched`, `plan`, `buffer`, `tensor`, `event`, `device`, `sync`

### `ggml/include/ggml-backend.h`

```text
24:     typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;
25:     typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
26:     typedef struct ggml_backend_event * ggml_backend_event_t;
28:     typedef void * ggml_backend_graph_plan_t;
30:     typedef struct ggml_backend_device * ggml_backend_dev_t;
34:     // Backend buffer type
37:     GGML_API const char *          ggml_backend_buft_name          (ggml_backend_buffer_type_t buft);
38:     GGML_API ggml_backend_buffer_t ggml_backend_buft_alloc_buffer  (ggml_backend_buffer_type_t buft, size_t size);
39:     GGML_API size_t                ggml_backend_buft_get_alignment (ggml_backend_buffer_type_t buft);
40:     GGML_API size_t                ggml_backend_buft_get_max_size  (ggml_backend_buffer_type_t buft);
41:     GGML_API size_t                ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor);
42:     GGML_API bool                  ggml_backend_buft_is_host       (ggml_backend_buffer_type_t buft);
43:     GGML_API ggml_backend_dev_t    ggml_backend_buft_get_device    (ggml_backend_buffer_type_t buft);
46:     // Backend buffer
49:     enum ggml_backend_buffer_usage {
50:         GGML_BACKEND_BUFFER_USAGE_ANY = 0,
51:         GGML_BACKEND_BUFFER_USAGE_WEIGHTS = 1,
52:         GGML_BACKEND_BUFFER_USAGE_COMPUTE = 2,
55:     GGML_API const char *                   ggml_backend_buffer_name          (ggml_backend_buffer_t buffer);
56:     GGML_API void                           ggml_backend_buffer_free          (ggml_backend_buffer_t buffer);
57:     GGML_API void *                         ggml_backend_buffer_get_base      (ggml_backend_buffer_t buffer);
58:     GGML_API size_t                         ggml_backend_buffer_get_size      (ggml_backend_buffer_t buffer);
59:     GGML_API enum ggml_status               ggml_backend_buffer_init_tensor   (ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
60:     GGML_API size_t                         ggml_backend_buffer_get_alignment (ggml_backend_buffer_t buffer);
61:     GGML_API size_t                         ggml_backend_buffer_get_max_size  (ggml_backend_buffer_t buffer);
62:     GGML_API size_t                         ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor);
63:     GGML_API void                           ggml_backend_buffer_clear         (ggml_backend_buffer_t buffer, uint8_t value);
64:     GGML_API bool                           ggml_backend_buffer_is_host       (ggml_backend_buffer_t buffer);
65:     GGML_API void                           ggml_backend_buffer_set_usage     (ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage);
66:     GGML_API enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage     (ggml_backend_buffer_t buffer);
67:     GGML_API ggml_backend_buffer_type_t     ggml_backend_buffer_get_type      (ggml_backend_buffer_t buffer);
68:     GGML_API void                           ggml_backend_buffer_reset         (ggml_backend_buffer_t buffer);
70:     // tensor copy between different backends
71:     GGML_API void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst);
81:     GGML_API ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend);
82:     GGML_API ggml_backend_buffer_t      ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size);
86:     GGML_API void ggml_backend_tensor_set_async   (ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
87:     GGML_API void ggml_backend_tensor_get_async   (ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
88:     GGML_API void ggml_backend_tensor_set_2d_async(ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
89:     GGML_API void ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
91:     // "offset" refers to the offset in tensor->data for setting/getting data
92:     GGML_API void ggml_backend_tensor_set   (      struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
93:     GGML_API void ggml_backend_tensor_get   (const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
94:     GGML_API void ggml_backend_tensor_set_2d(      struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
95:     GGML_API void ggml_backend_tensor_get_2d(const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
96:     GGML_API void ggml_backend_tensor_memset(      struct ggml_tensor * tensor,     uint8_t value, size_t offset, size_t size);
98:     GGML_API void ggml_backend_synchronize(ggml_backend_t backend);
100:     GGML_API ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph);
101:     GGML_API void                      ggml_backend_graph_plan_free  (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
103:     GGML_API enum ggml_status ggml_backend_graph_plan_compute (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
104:     GGML_API enum ggml_status ggml_backend_graph_compute      (ggml_backend_t backend, struct ggml_cgraph * cgraph);
105:     GGML_API enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph);
107:     // NOTE: will be removed, use device version instead
108:     GGML_API bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op);
109:     GGML_API bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft);
110:     GGML_API bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op);
112:     // asynchronous copy
115:     // automatic fallback to sync copy if async is not supported
116:     GGML_API void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);
118:     GGML_API ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend);
121:     // Events
124:     GGML_API ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device);
125:     GGML_API void                 ggml_backend_event_free(ggml_backend_event_t event);
126:     GGML_API void                 ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend);
127:     GGML_API void                 ggml_backend_event_synchronize(ggml_backend_event_t event);
128:     GGML_API void                 ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event);
131:     // Backend device
135:         // CPU device using system memory
136:         GGML_BACKEND_DEVICE_TYPE_CPU,
137:         // GPU device using dedicated memory
138:         GGML_BACKEND_DEVICE_TYPE_GPU,
139:         // integrated GPU device using host memory
140:         GGML_BACKEND_DEVICE_TYPE_IGPU,
141:         // accelerator devices intended to be used together with the CPU backend (e.g. BLAS or AMX)
142:         GGML_BACKEND_DEVICE_TYPE_ACCEL,
143:         // "meta" device wrapping multiple other devices for tensor parallelism
144:         GGML_BACKEND_DEVICE_TYPE_META,
147:     // functionality supported by the device
149:         // asynchronous operations
150:         bool async;
151:         // pinned host buffer
152:         bool host_buffer;
153:         // creating buffers from host ptr
154:         bool buffer_from_host_ptr;
155:         // event synchronization
156:         bool events;
159:     // all the device properties
161:         // device name
163:         // device description
165:         // device free memory in bytes
167:         // device total memory in bytes
169:         // device type
171:         // device id
172:         //   for PCI devices, this should be the PCI bus id formatted as "domain:bus:device.function" (e.g. "0000:01:00.0")
174:         const char * device_id;
175:         // device capabilities
179:     GGML_API const char *                  ggml_backend_dev_name(ggml_backend_dev_t device);
180:     GGML_API const char *                  ggml_backend_dev_description(ggml_backend_dev_t device);
181:     GGML_API void                          ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total);
182:     GGML_API enum ggml_backend_dev_type    ggml_backend_dev_type(ggml_backend_dev_t device);
183:     GGML_API void                          ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props);
184:     GGML_API ggml_backend_reg_t            ggml_backend_dev_backend_reg(ggml_backend_dev_t device);
185:     GGML_API ggml_backend_t                ggml_backend_dev_init(ggml_backend_dev_t device, const char * params);
186:     GGML_API ggml_backend_buffer_type_t    ggml_backend_dev_buffer_type(ggml_backend_dev_t device);
187:     GGML_API ggml_backend_buffer_type_t    ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device);
188:     GGML_API ggml_backend_buffer_t         ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size);
190:     GGML_API bool                          ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op);
191:     GGML_API bool                          ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft);
192:     GGML_API bool                          ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op);
205:     // Context management and operations for faster communication between backends, used for tensor parallelism (meta backend)
208:     typedef bool   (*ggml_backend_comm_allreduce_tensor_t)(void * comm_ctx, struct ggml_tensor ** tensors);
210:     // Split buffer type for tensor parallelism (old)
211:     typedef ggml_backend_buffer_type_t   (*ggml_backend_split_buffer_type_t)(int main_device, const float * tensor_split);
214:     // Get additional buffer types provided by the device (returns a NULL-terminated array)
215:     typedef ggml_backend_buffer_type_t * (*ggml_backend_dev_get_extra_bufts_t)(ggml_backend_dev_t device);
231:     GGML_API void ggml_backend_device_register(ggml_backend_dev_t device);
238:     // Device enumeration
261:     // Backend scheduler
264:     // The backend scheduler allows for multiple backend devices to be used together
265:     // Handles compute buffer allocation, assignment of tensors to backends, and copying of tensors between backends
268:     // - the location of the pre-allocated tensors (e.g. the weights)
272:         // operations that use tensors allocated in a buffer with USAGE_WEIGHTS will be assigned
273:         // preferably to run on the same backend as the buffer
274:         ggml_backend_buffer_set_usage(buf_weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
276:         sched = ggml_backend_sched_new({backend_gpu, backend_gpu2, backend_cpu}, NULL, num_backends, GGML_DEFAULT_GRAPH_SIZE, false, true);
278:         // initialize buffers from a max size graph (optional)
279:         reserve_graph = build_graph(sched, max_batch_size);
282:         struct ggml_tensor * node = ggml_mul_mat(ctx, ...);
283:         ggml_backend_sched_set_tensor_backend(sched, node, backend_gpu);
285:         ggml_backend_sched_reserve(sched, reserve_graph);
287:         // compute
288:         graph = build_graph(sched); // the graph and its tensors are single-use in terms of allocation, multi-use in terms of computation
290:             ggml_backend_sched_graph_compute(sched, graph); // on the first iteration the graph is allocated automatically
293:         // if there are graph inputs:
294:         graph = build_graph(sched); // get a new graph that is not allocated (the metadata for the old graph is freed once ggml_free is called)
295:         ggml_backend_sched_reset(sched); // clear the allocation of the previous graph
296:         ggml_backend_sched_alloc_graph(sched, graph); // explicitly allocate the new graph but do not execute it
297:         ggml_backend_tensor_set(input_tensor, ...); // copy data to the newly allocated graph tensors
298:         ggml_backend_sched_graph_compute(sched, graph); // execute the graph
301:         // allocate them statically via ggml_backend_alloc_ctx_tensors
305:     typedef struct ggml_backend_sched * ggml_backend_sched_t;
307:     // Evaluation callback for each node in the graph (set with ggml_backend_sched_set_eval_callback)
308:     // when ask == true, the scheduler wants to know if the user wants to observe this node
309:     // this allows the scheduler to batch nodes together in order to evaluate them in a single call
311:     // when ask == false, the scheduler is passing the node tensor to the user for observation
312:     // if the user returns false, the scheduler will cancel the graph compute
314:     typedef bool (*ggml_backend_sched_eval_callback)(struct ggml_tensor * t, bool ask, void * user_data);
316:     // Initialize a backend scheduler, backends with low index are given priority over backends with high index
317:     GGML_API ggml_backend_sched_t ggml_backend_sched_new(ggml_backend_t * backends, ggml_backend_buffer_type_t * bufts, int n_backends, size_t graph_size, bool parallel, bool op_offload);
318:     GGML_API void                 ggml_backend_sched_free(ggml_backend_sched_t sched);
320:     // Initialize backend buffers from a measure graph
321:     GGML_API void                 ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes);
322:     GGML_API bool                 ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph); // returns success
324:     GGML_API int                  ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched);
325:     GGML_API ggml_backend_t       ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i);
327:     // Get the number of splits of the last graph
328:     GGML_API int                  ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched);
329:     GGML_API int                  ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched);
331:     GGML_API ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend);
332:     GGML_API size_t                     ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend);
334:     GGML_API void                 ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend);
335:     GGML_API ggml_backend_t       ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node);
337:     // Split graph without allocating it
338:     GGML_API void                 ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
340:     // Allocate and compute graph on the backend scheduler
341:     GGML_API bool                 ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph); // returns success
342:     GGML_API enum ggml_status     ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
343:     GGML_API enum ggml_status     ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
344:     GGML_API void                 ggml_backend_sched_synchronize(ggml_backend_sched_t sched);
346:     // Reset all assignments and allocators - must be called before changing the node backends or allocating a new graph.
347:     // This in effect deallocates all tensors that were previously allocated and leaves them with dangling pointers.
348:     // The correct way to use this API is to discard the deallocated tensors and create new ones.
349:     GGML_API void                 ggml_backend_sched_reset(ggml_backend_sched_t sched);
351:     // Set a callback to be called for each resulting node during graph compute
352:     GGML_API void                 ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data);
358: #define GGML_BACKEND_META_MAX_DEVICES 16
361:         // tensor split by tensor dimensions:
379:         // for tensors with axis >= 0 && axis < GGML_MAX_DIMS:
380:         //   - each device has a slice of the tensor along the split axis
381:         //   - most tensors have n_segments == 1 and a contiguous slice of the tensor data
382:         //   - some tensors have an inhomogenenous data layout along the split axis,
383:         //     those tensors are divided into segments which are each individually split across devices
384:         //   - ne has one entry per segment and device that add up to ggml_tensor::ne for that axis,
385:         //     the outer/inner loops are over segments/devices like [seg0_dev0, seg0_dev1, seg1_dev0, seg1_dev1],
387:         //     that each need to be split individually across devices so that each device gets a slice of Q, K, and V
388:         int64_t  ne[16*GGML_BACKEND_META_MAX_DEVICES];
392:     // function to assign split states for statically allocated tensors, compute tensor split states will be assigned to be compatible:
393:     typedef struct ggml_backend_meta_split_state(*ggml_backend_meta_get_split_state_t)(const struct ggml_tensor * tensor, void * userdata);
395:     // create a new meta device from "simple" devices, meta buffer type/buffer/backend is then derived from this:
396:     // TODO: this looks a bit strange - a backend API creates a device. I think we should try
398:     GGML_API ggml_backend_dev_t ggml_backend_meta_device(
405:     struct ggml_backend_graph_copy {
406:         ggml_backend_buffer_t buffer;
409:         struct ggml_cgraph * graph;
412:     // Copy a graph to a different backend
413:     GGML_API struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph);
414:     GGML_API void                           ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy);
416:     typedef bool (*ggml_backend_eval_callback)(int node_index, struct ggml_tensor * t1, struct ggml_tensor * t2, void * user_data);
419:     GGML_API bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor const * const * test_nodes, size_t num_test_nodes);
421:     // Tensor initialization
422:     GGML_API enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr);
423:     GGML_API enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor);
425:     // CPU buffer types are always available
426:     GGML_API ggml_backend_buffer_t      ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size);
427:     GGML_API ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void);
```

### `ggml/src/ggml-backend-impl.h`

```text
14:     // Backend buffer type
17:     struct ggml_backend_buffer_type_i {
18:         const char *          (*get_name)      (ggml_backend_buffer_type_t buft);
19:         // allocate a buffer of this type
20:         ggml_backend_buffer_t (*alloc_buffer)  (ggml_backend_buffer_type_t buft, size_t size);
21:         // tensor alignment
22:         size_t                (*get_alignment) (ggml_backend_buffer_type_t buft);
23:         // (optional) max buffer size that can be allocated (defaults to SIZE_MAX)
24:         size_t                (*get_max_size)  (ggml_backend_buffer_type_t buft);
25:         // (optional) data size needed to allocate the tensor, including padding (defaults to ggml_nbytes)
26:         size_t                (*get_alloc_size)(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor);
27:         // (optional) check if tensor data is in host memory and uses standard ggml tensor layout (defaults to false)
28:         bool                  (*is_host)       (ggml_backend_buffer_type_t buft);
31:     struct ggml_backend_buffer_type {
32:         struct ggml_backend_buffer_type_i  iface;
33:         ggml_backend_dev_t device;
38:     // Backend buffer
41:     struct ggml_backend_buffer_i {
42:         // (optional) free the buffer
43:         void         (*free_buffer)  (ggml_backend_buffer_t buffer);
44:         // base address of the buffer
45:         void *       (*get_base)     (ggml_backend_buffer_t buffer);
46:         // (optional) initialize a tensor in the buffer (eg. add tensor extras)
47:         enum ggml_status (*init_tensor)(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
48:         // tensor data access
49:         void         (*memset_tensor)(ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor,     uint8_t value, size_t offset, size_t size);
50:         void         (*set_tensor)   (ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
51:         void         (*get_tensor)   (ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
53:         void         (*set_tensor_2d)(ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
54:         void         (*get_tensor_2d)(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
56:         // (optional) tensor copy: dst is in the buffer, src may be in any buffer, including buffers from a different backend (return false if not supported)
57:         bool         (*cpy_tensor)   (ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst);
58:         // clear the entire buffer
59:         void         (*clear)        (ggml_backend_buffer_t buffer, uint8_t value);
60:         // (optional) reset any internal state due to tensor initialization, such as tensor extras
61:         void         (*reset)        (ggml_backend_buffer_t buffer);
64:     struct ggml_backend_buffer {
65:         struct ggml_backend_buffer_i  iface;
66:         ggml_backend_buffer_type_t    buft;
69:         enum ggml_backend_buffer_usage usage;
72:     GGML_API ggml_backend_buffer_t ggml_backend_buffer_init(
73:                    ggml_backend_buffer_type_t buft,
74:             struct ggml_backend_buffer_i      iface,
78:     // do not use directly, use ggml_backend_tensor_copy instead
79:     GGML_API bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst);
81:     // multi-buffer
82:     // buffer that contains a collection of buffers
83:     GGML_API ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers);
84:     GGML_API bool                  ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer);
85:     GGML_API void                  ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage);
92:     GGML_API bool ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf);
93:     GGML_API bool ggml_backend_buft_is_meta  (ggml_backend_buffer_type_t buft);
98:     // temporary workaround to statically allocate tensors from a context in a deduplicated way:
99:     GGML_API struct ggml_backend_buffer * ggml_backend_meta_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft);
110:         // (optional) asynchronous tensor data access
111:         void (*set_tensor_async)   (ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
112:         void (*get_tensor_async)   (ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
113:         void (*set_tensor_2d_async)(ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
114:         void (*get_tensor_2d_async)(ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
115:         bool (*cpy_tensor_async)(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);
117:         // (optional) complete all pending operations (required if the backend supports async operations)
118:         void (*synchronize)(ggml_backend_t backend);
120:         // (optional) graph plans (not used currently)
121:         // compute graph with a plan
122:         ggml_backend_graph_plan_t (*graph_plan_create) (ggml_backend_t backend, const struct ggml_cgraph * cgraph);
123:         void                      (*graph_plan_free)   (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
124:         // update the plan with a new graph - this should be faster than creating a new plan when the graph has the same topology
125:         void                      (*graph_plan_update) (ggml_backend_t backend, ggml_backend_graph_plan_t plan, const struct ggml_cgraph * cgraph);
126:         // compute the graph with the plan
127:         enum ggml_status          (*graph_plan_compute)(ggml_backend_t backend, ggml_backend_graph_plan_t plan);
129:         // compute graph (always async if supported by the backend)
130:         enum ggml_status          (*graph_compute)     (ggml_backend_t backend, struct ggml_cgraph * cgraph);
132:         // (optional) event synchronization
133:         // record an event on this stream
134:         void (*event_record)(ggml_backend_t backend, ggml_backend_event_t event);
135:         // wait for an event on on a different stream
136:         void (*event_wait)  (ggml_backend_t backend, ggml_backend_event_t event);
138:         // (optional) sort/optimize the nodes in the graph
139:         void                      (*graph_optimize)    (ggml_backend_t backend, struct ggml_cgraph * cgraph);
145:         ggml_backend_dev_t device;
149:     struct ggml_backend_event {
150:         struct ggml_backend_device * device;
155:     // Backend device
160:     struct ggml_backend_device_i {
161:         // device name: short identifier for this device, such as "CPU" or "CUDA0"
164:         // device description: short informative description of the device, could be the model name
167:         // device memory in bytes: 0 bytes to indicate no memory to report
170:         // device type
173:         // device properties
179:         // preferred buffer type
180:         ggml_backend_buffer_type_t (*get_buffer_type)(ggml_backend_dev_t dev);
182:         // (optional) host buffer type (in system memory, typically this is a pinned memory buffer for faster transfers between host and device)
183:         ggml_backend_buffer_type_t (*get_host_buffer_type)(ggml_backend_dev_t dev);
185:         // (optional) buffer from pointer: create a buffer from a host pointer (useful for memory mapped models and importing data from other libraries)
186:         ggml_backend_buffer_t (*buffer_from_host_ptr)(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size);
188:         // check if the backend can compute an operation
189:         bool (*supports_op)(ggml_backend_dev_t dev, const struct ggml_tensor * op);
191:         // check if the backend can use tensors allocated in a buffer type
192:         bool (*supports_buft)(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft);
194:         // (optional) check if the backend wants to run an operation, even if the weights are allocated in an incompatible buffer
196:         bool (*offload_op)(ggml_backend_dev_t dev, const struct ggml_tensor * op);
198:         // (optional) event synchronization
199:         ggml_backend_event_t (*event_new)         (ggml_backend_dev_t dev);
200:         void                 (*event_free)        (ggml_backend_dev_t dev, ggml_backend_event_t event);
201:         void                 (*event_synchronize) (ggml_backend_dev_t dev, ggml_backend_event_t event);
204:     struct ggml_backend_device {
205:         struct ggml_backend_device_i iface;
217:         // enumerate available devices
218:         size_t             (*get_device_count)(ggml_backend_reg_t reg);
219:         ggml_backend_dev_t (*get_device)(ggml_backend_reg_t reg, size_t index);
```

## G14A interpretation checklist

Use this report to decide the next step:

```text
1. If ggml_backend_i has graph/compute callbacks, create a compile-only G14B probe
   that populates exactly those fields with stubs matching this checkout.
2. If graph compute lives elsewhere, e.g. backend device or scheduler structs,
   inspect the listed raw structs before wiring any callback.
3. Do not assume newer upstream fields such as get_default_buffer_type exist in
   ggml_backend_i; this checkout already proved that assumption false in G12A.
4. Keep G13 backend_dispatch_op as the stable fallback until the real interface
   shape has a compile-checked adapter.
```

