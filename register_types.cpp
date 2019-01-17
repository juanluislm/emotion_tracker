#include "register_types.h"
#include "emotion_tracker.h"

void register_emotion_tracker_types()
{
		ClassDB::register_type<EmotionTracker>();
}

void unregister_emotion_tracker_types() {
   //nothing to do here
}
