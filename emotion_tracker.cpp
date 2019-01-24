#include "emotion_tracker.h"

EmotionTracker::EmotionTracker() {

}

bool EmotionTracker::_config(const string & model_path) {
  settings.model_name = model_path;
  model = tflite::FlatBufferModel::BuildFromFile(settings.model_name.c_str());
  if (!model) {
    LOG(FATAL) << "\nFailed to mmap model " << settings.model_name << "\n";
    exit(-1);
  }
  LOG(INFO) << "Loaded model " << settings.model_name << "\n";
  model->error_reporter();
  LOG(INFO) << "resolved reporter\n";

  tflite::ops::builtin::BuiltinOpResolver resolver;

  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
    LOG(FATAL) << "Failed to construct interpreter\n";
    exit(-1);
  }
  // only makes sense if Android NNAPI is available
  interpreter->UseNNAPI(settings.accel);

  if (settings.number_of_threads != -1) {
    interpreter->SetNumThreads(settings.number_of_threads);
  }

  if (settings.verbose) {
    LOG(INFO) << "tensors size: " << interpreter->tensors_size() << "\n";
    LOG(INFO) << "nodes size: " << interpreter->nodes_size() << "\n";
    LOG(INFO) << "inputs: " << interpreter->inputs().size() << "\n";
    LOG(INFO) << "input(0) name: " << interpreter->GetInputName(0) << "\n";

    int t_size = interpreter->tensors_size();
    for (int i = 0; i < t_size; i++) {
      if (interpreter->tensor(i)->name)
        LOG(INFO) << i << ": " << interpreter->tensor(i)->name << ", "
                  << interpreter->tensor(i)->bytes << ", "
                  << interpreter->tensor(i)->type << ", "
                  << interpreter->tensor(i)->params.scale << ", "
                  << interpreter->tensor(i)->params.zero_point << "\n";
    }
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(FATAL) << "Failed to allocate tensors!";
  }

  // get input dimension from the input tensor metadata
  // assuming one input only
  input = interpreter->inputs()[0];
  TfLiteIntArray* dims = interpreter->tensor(input)->dims;
  settings.wanted_height = dims->data[1];
  settings.wanted_width = dims->data[2];
  settings.wanted_channels = dims->data[3];

  if (interpreter->tensor(input)->type == kTfLiteFloat32) {
    settings.input_floating = true;
  } else if (interpreter->tensor(input)->type == kTfLiteUInt8) {
    settings.input_floating = false;
  } else {
    LOG(FATAL) << "cannot handle input type "
               << interpreter->tensor(input)->type << " yet";
    exit(-1);
  }
  // get output dimensions from the output tensor metadata
  output = interpreter->outputs()[0];
  if (interpreter->tensor(output)->type == kTfLiteFloat32) {
    settings.output_floating = true;
  } else if (interpreter->tensor(output)->type == kTfLiteUInt8) {
    settings.output_floating = false;
  } else {
    LOG(FATAL) << "cannot handle output type "
               << interpreter->tensor(input)->type << " yet";
    exit(-1);
  }

  output_size = 7;

  const char* args[] = {"angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"};
  labels = std::vector<string>(args, args + 7);

  emotionResult.clear();
  for (size_t i = 0; i < output_size; i++) {
    emotionResult.push_back(0);
  }
#ifdef DEBUGCV
    timer = Timer("EmotionTracker");
#endif
  return true;
}

void EmotionTracker::track(uint8_t * in, int image_width, int image_height, int image_channels) {
  #ifdef DEBUGCV
    timer.start_timer();
  #endif
  if (settings.input_floating) {
    resize<float>(interpreter->typed_tensor<float>(input), in,
                  image_height, image_width, image_channels,
                  &settings);
  } else {
    resize<uint8_t>(interpreter->typed_tensor<uint8_t>(input), in,
                    image_height, image_width, image_channels,
                    &settings);
  }
  #ifdef DEBUGCV
    timer.update_timer("preprocessing input");
  #endif
  for (int i = 0; i < settings.loop_count; i++) {
    if (interpreter->Invoke() != kTfLiteOk) {
      LOG(FATAL) << "Failed to invoke tflite!\n";
    }
  }
  #ifdef DEBUGCV
    timer.update_timer("forward path");
  #endif
  int likely_emotion=0;
  float max_emotion = 0.0;
  for (int i = 0; i < output_size; i++) {
    if (settings.output_floating) {
      emotionResult[i] = interpreter->typed_output_tensor<float>(0)[i];
    } else {
      emotionResult[i] = interpreter->typed_output_tensor<uint8_t>(0)[i] / 255.0;
    }

    if(emotionResult[i] > max_emotion ){
      max_emotion = emotionResult[i];
      likely_emotion = i;
    }
  }

  likely_emotion_idx   = likely_emotion;
  likely_emotion_label = labels[ likely_emotion_idx ];
}

std::vector<string> EmotionTracker::get_labels() {
  return labels;
}

std::vector<float> EmotionTracker::get_results() {
  return emotionResult;
}

int EmotionTracker::get_output_size() {
  return output_size;
}

int EmotionTracker::get_likely_emotion(){

  return likely_emotion_idx;
}

string EmotionTracker::get_likely_emotion_label(){
  return likely_emotion_label;
}

void EmotionTracker::_bind_methods(){
  ClassDB::bind_method(D_METHOD("get_likely_emotion"), &EmotionTracker::get_likely_emotion);

  ClassDB::bind_method(D_METHOD("get_likely_emotion_label"), &EmotionTracker::get_likely_emotion_label);

  ClassDB::bind_method(D_METHOD("get_output_size"), &EmotionTracker::get_output_size);

  ClassDB::bind_method(D_METHOD("get_results"), &EmotionTracker::get_results);

  ClassDB::bind_method(D_METHOD("get_labels"), &EmotionTracker::get_labels);

  ClassDB::bind_method(D_METHOD("track"), "in", "image_width", 
                            "image_height", "image_channels", 
                            &EmotionTracker::track);
  // in, int image_width, int image_height, int image_channels
}
