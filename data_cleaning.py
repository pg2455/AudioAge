import os
import soundfile as sf
import sounddevice as sd

# Convert mp3 to wav
def covert_all():
    from pydub import AudioSegment
    import ffmpy
    DATADIR="../Common_voice/Common_Voice_train"
    for gender in ['male', 'female']:
        for task_type in ['ages', 'accents']:
            for class_dir in os.listdir(os.path.join(DATADIR, gender, task_type)):
                if class_dir == ".DS_Store" or class_dir == "male_fifties":
                    continue
                for mp3_sound_file in os.listdir(os.path.join(DATADIR, gender, task_type, class_dir)):
                    filename=os.path.join(DATADIR, gender, task_type, class_dir, mp3_sound_file)
                    ff = ffmpy.FFmpeg(
                        inputs={filename:None},
                        outputs={filename[0:-4] + ".wav": None}
                    )
                    ff.run()
                    os.remove(filename)
