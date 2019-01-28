#pragma once
#include "timer.h"
// #include <string>
#include <vector>

#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/string_util.h"
#include "tensorflow/contrib/lite/version.h"
#include "tensorflow/core/public/version.h"

#include "core/io/resource_loader.h"
#include "core/os/thread.h"
#include "core/reference.h"
#include "core/resource.h"
#include "core/array.h"
#include "core/int_types.h"

#define LOG(x) std::cerr

using namespace tflite;

struct EmotionTFSettings {
  bool verbose = false;
  // whether to use Android NNAPI for hardware accelaration
  bool accel = false;
  bool input_floating = false;
  bool output_floating = false;
  int loop_count = 1;
  float input_mean = 127.5f;
  float input_std = 127.5f;
  string model_name = "../models/emotion_mini_XCEPTION_64x64_0.66_7ms.hdf5.pb.tflite";
  string input_layer_type = "uint8_t";
  int wanted_width;
  int wanted_height;
  int wanted_channels;
  int number_of_threads = 4;
};

template <class T>
void resize(T* out, uint8_t* in, int image_height, int image_width, int image_channels, EmotionTFSettings* s) {
  int number_of_pixels = image_height * image_width * image_channels;
  std::unique_ptr<Interpreter> interpreter(new Interpreter);

  int base_index = 0;

  // two inputs: input and new_sizes
  interpreter->AddTensors(2, &base_index);
  // one output
  interpreter->AddTensors(1, &base_index);
  // set input and output tensors
  interpreter->SetInputs({0, 1});
  interpreter->SetOutputs({2});

  // set parameters of tensors
  TfLiteQuantizationParams quant;
  interpreter->SetTensorParametersReadWrite(
      0, kTfLiteFloat32, "input",
      {1, image_height, image_width, image_channels}, quant);
  interpreter->SetTensorParametersReadWrite(1, kTfLiteInt32, "new_size", {2},
                                            quant);
  interpreter->SetTensorParametersReadWrite(
      2, kTfLiteFloat32, "output",
      {1, s->wanted_height, s->wanted_width, s->wanted_channels}, quant);

  // TODO: Check with tf folks how to get the appropriate value for the extra parameter
  // inputing 1 just to keep going.
  // JM

  ops::builtin::BuiltinOpResolver resolver;

  #if TF_MINOR_VERSION >= 10
    const TfLiteRegistration* resize_op = resolver.FindOp(BuiltinOperator_RESIZE_BILINEAR,1);
  #else
    const TfLiteRegistration* resize_op = resolver.FindOp(BuiltinOperator_RESIZE_BILINEAR);
  #endif


  auto* params = reinterpret_cast<TfLiteResizeBilinearParams*>(
      malloc(sizeof(TfLiteResizeBilinearParams)));
  params->align_corners = false;
  interpreter->AddNodeWithParameters({0, 1}, {2}, nullptr, 0, params, resize_op,
                                     nullptr);

  interpreter->AllocateTensors();

  // fill input image
  // in[] are integers, cannot do memcpy() directly
  auto input = interpreter->typed_tensor<float>(0);
  for (int i = 0; i < number_of_pixels; i++) {
    input[i] = in[i];
  }

  // fill new_sizes
  interpreter->typed_tensor<int>(1)[0] = s->wanted_height;
  interpreter->typed_tensor<int>(1)[1] = s->wanted_width;

  interpreter->Invoke();

  auto output = interpreter->typed_tensor<float>(2);
  auto output_number_of_pixels =
    (s->wanted_height) * (s->wanted_height) * (s->wanted_channels);

  for (int i = 0; i < output_number_of_pixels; i++) {
    if (s->input_floating)
      out[i] = (output[i] - s->input_mean) / s->input_std;
    else
      out[i] = (uint8_t)output[i];
  }
}


class EmotionTracker : public Reference {

  GDCLASS(EmotionTracker, Reference);
  
public:
  EmotionTracker();

  bool _config(string model_path);
  
    

  void track(uint8_t * in, int image_width, int image_height, int image_channels);
  
  EmotionTFSettings settings;
  int output_size;
  std::vector<string> labels;
  std::vector<float> emotionResult;

  std::vector<string> get_labels();

  std::vector<float> get_results();

  int get_output_size();

  int get_likely_emotion();

  string get_likely_emotion_label();
  
  bool g_config(String model_path);
  
    String g_get_likely_emotion_label();
    
    Array g_get_results();
    
    Array g_get_labels();
    

protected:

  static void _bind_methods();

private:

  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
  string likely_emotion_label;
  int likely_emotion_idx;
  // input / output node index
  int input;
  int output;
#ifdef DEBUGCV
  Timer timer;
#endif
};
