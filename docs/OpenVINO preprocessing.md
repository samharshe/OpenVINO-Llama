Preprocessing
This guide introduces how preprocessing works in API 2.0 by a comparison with preprocessing in the previous Inference Engine API. It also demonstrates how to migrate preprocessing scenarios from Inference Engine to API 2.0 via code samples.

How Preprocessing Works in API 2.0
Inference Engine API contains preprocessing capabilities in the InferenceEngine::CNNNetwork class. Such preprocessing information is not a part of the main inference graph executed by OpenVINO devices. Therefore, it is stored and executed separately before the inference stage:

Preprocessing operations are executed on the CPU for most OpenVINO inference plugins. Thus, instead of occupying accelerators, they keep the CPU busy with computational tasks.

Preprocessing information stored in InferenceEngine::CNNNetwork is lost when saving back to the OpenVINO IR file format.

API 2.0 introduces a new way of adding preprocessing operations to the model - each preprocessing or post-processing operation is integrated directly into the model and compiled together with the inference graph:

API 2.0 first adds preprocessing operations by using ov::preprocess::PrePostProcessor,

and then compiles the model on the target by using ov::Core::compile_model.

Having preprocessing operations as a part of an OpenVINO opset makes it possible to read and serialize a preprocessed model as the OpenVINO™ IR file format.

More importantly, API 2.0 does not assume any default layouts as Inference Engine did. For example, both { 1, 224, 224, 3 } and { 1, 3, 224, 224 } shapes are supposed to be in the NCHW layout, while only the latter is. Therefore, some preprocessing capabilities in the API require layouts to be set explicitly. To learn how to do it, refer to the Layout overview. For example, to perform image scaling by partial dimensions H and W, preprocessing needs to know what dimensions H and W are.

Note

Use model conversion API preprocessing capabilities to insert preprocessing operations in your model for optimization. Thus, the application does not need to read the model and set preprocessing repeatedly. You can use the model caching feature to improve the time-to-inference.

The following sections demonstrate how to migrate preprocessing scenarios from Inference Engine API to API 2.0. The snippets assume that you need to preprocess a model input with the tensor_name in Inference Engine API, using operation_name to address the data.

Preparation: Import Preprocessing in Python
In order to utilize preprocessing, the following imports must be added.

Inference Engine API

from openvino import Core, Layout, Type
from openvino.preprocess import ColorFormat, PrePostProcessor, ResizeAlgorithm
API 2.0

from openvino import Core, Layout, Type
from openvino.preprocess import ColorFormat, PrePostProcessor, ResizeAlgorithm
There are two different namespaces:
* runtime, which contains API 2.0 classes;
* and preprocess, which provides Preprocessing API.
Using Mean and Scale Values
Inference Engine API


Python

C++

C
    // No preprocessing related interfaces provided by C API 1.0
API 2.0


Python

C++

C
    ov_preprocess_prepostprocessor_t* preprocess = NULL;
    ov_preprocess_input_info_t* input_info = NULL;
    ov_preprocess_input_model_info_t* input_model = NULL;
    ov_preprocess_preprocess_steps_t* input_process = NULL;
    ov_layout_t* layout = NULL;
    ov_model_t* new_model = NULL;

    ov_preprocess_prepostprocessor_create(model, &preprocess);
    ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info);
    ov_preprocess_input_info_get_model_info(input_info, &input_model);
    // we only need to know where is C dimension
    ov_layout_create("...C", &layout);
    ov_preprocess_input_model_info_set_layout(input_model, layout);
    // specify scale and mean values, order of operations is important
    ov_preprocess_input_info_get_preprocess_steps(input_info, &input_process);
    ov_preprocess_preprocess_steps_mean(input_process, 116.78f);
    ov_preprocess_preprocess_steps_scale(input_process, 57.21f);
    // insert preprocessing operations to the 'model'
    ov_preprocess_prepostprocessor_build(preprocess, &new_model);

    ov_layout_free(layout);
    ov_model_free(new_model);
    ov_preprocess_input_model_info_free(input_model);
    ov_preprocess_preprocess_steps_free(input_process);
    ov_preprocess_input_info_free(input_info);
    ov_preprocess_prepostprocessor_free(preprocess);
Converting Precision and Layout
Inference Engine API


Python

C++

C
    // No preprocessing related interfaces provided by C API 1.0
API 2.0


Python

C++

C
    ov_preprocess_prepostprocessor_t* preprocess = NULL;
    ov_preprocess_input_info_t* input_info = NULL;
    ov_preprocess_input_tensor_info_t* input_tensor_info = NULL;
    ov_layout_t* layout_nhwc = NULL;
    ov_preprocess_input_model_info_t* input_model = NULL;
    ov_layout_t* layout_nchw = NULL;
    ov_model_t* new_model = NULL;

    ov_preprocess_prepostprocessor_create(model, &preprocess);
    ov_preprocess_prepostprocessor_get_input_info_by_name(preprocess, tensor_name, &input_info);
    ov_preprocess_input_info_get_tensor_info(input_info, &input_tensor_info);

    ov_layout_create("NHWC", &layout_nhwc);
    ov_preprocess_input_tensor_info_set_layout(input_tensor_info, layout_nhwc);
    ov_preprocess_input_tensor_info_set_element_type(input_tensor_info, ov_element_type_e::U8);

    ov_preprocess_input_info_get_model_info(input_info, &input_model);
    ov_layout_create("NCHW", &layout_nchw);
    ov_preprocess_input_model_info_set_layout(input_model, layout_nchw);
    // layout and precision conversion is inserted automatically,
    // because tensor format != model input format
    ov_preprocess_prepostprocessor_build(preprocess, &new_model);

    ov_layout_free(layout_nchw);
    ov_layout_free(layout_nhwc);
    ov_model_free(new_model);
    ov_preprocess_input_model_info_free(input_model);
    ov_preprocess_input_tensor_info_free(input_tensor_info);
    ov_preprocess_input_info_free(input_info);
    ov_preprocess_prepostprocessor_free(preprocess);
Using Image Scaling
Inference Engine API


Python

C++

C
    // No preprocessing related interfaces provided by C API 1.0
API 2.0


Python

C++

C
    ov_preprocess_prepostprocessor_t* preprocess = NULL;
    ov_preprocess_input_info_t* input_info = NULL;
    ov_preprocess_input_tensor_info_t* input_tensor_info = NULL;
    ov_preprocess_input_model_info_t* input_model = NULL;
    ov_layout_t* layout = NULL;
    ov_preprocess_preprocess_steps_t* input_process = NULL;
    ov_model_t* new_model = NULL;

    ov_preprocess_prepostprocessor_create(model, &preprocess);
    ov_preprocess_prepostprocessor_get_input_info_by_name(preprocess, tensor_name, &input_info);
    ov_preprocess_input_info_get_tensor_info(input_info, &input_tensor_info);
    // scale from the specified tensor size
    ov_preprocess_input_tensor_info_set_spatial_static_shape(input_tensor_info, 448, 448);
    // need to specify H and W dimensions in model, others are not important
    ov_preprocess_input_info_get_model_info(input_info, &input_model);
    ov_layout_create("??HW", &layout);
    ov_preprocess_input_model_info_set_layout(input_model, layout);
    // scale to model shape
    ov_preprocess_input_info_get_preprocess_steps(input_info, &input_process);
    ov_preprocess_preprocess_steps_resize(input_process, ov_preprocess_resize_algorithm_e::RESIZE_LINEAR);
    // and insert operations to the model
    ov_preprocess_prepostprocessor_build(preprocess, &new_model);

    ov_layout_free(layout);
    ov_preprocess_preprocess_steps_free(input_process);
    ov_preprocess_input_model_info_free(input_model);
    ov_preprocess_input_tensor_info_free(input_tensor_info);
    ov_preprocess_input_info_free(input_info);
    ov_model_free(new_model);
    ov_preprocess_prepostprocessor_free(preprocess);
Converting Color Space
API 2.0


Python

C++

C
    ov_preprocess_prepostprocessor_t* preprocess = NULL;
    ov_preprocess_input_info_t* input_info = NULL;
    ov_preprocess_input_tensor_info_t* input_tensor_info = NULL;
    ov_preprocess_preprocess_steps_t* input_process = NULL;
    ov_model_t* new_model = NULL;

    ov_preprocess_prepostprocessor_create(model, &preprocess);
    ov_preprocess_prepostprocessor_get_input_info_by_name(preprocess, tensor_name, &input_info);
    ov_preprocess_input_info_get_tensor_info(input_info, &input_tensor_info);
    ov_preprocess_input_tensor_info_set_color_format(input_tensor_info, ov_color_format_e::NV12_TWO_PLANES);
    // add NV12 to BGR conversion
    ov_preprocess_input_info_get_preprocess_steps(input_info, &input_process);
    ov_preprocess_preprocess_steps_convert_color(input_process, ov_color_format_e::BGR);
    // and insert operations to the model
    ov_preprocess_prepostprocessor_build(preprocess, &new_model);

    ov_preprocess_input_tensor_info_free(input_tensor_info);
    ov_preprocess_preprocess_steps_free(input_process);
    ov_preprocess_input_info_free(input_info);
    ov_preprocess_prepostprocessor_free(preprocess);
    ov_model_free(new_model);