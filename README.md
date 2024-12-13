# Real-time Image classification using an RTSP stream

I developed this Tensorflow Keras project to solve a simple use case: notifying Home Assistant (HA) if my son is in or out of his crib. From there I have HA switch off the night light and change the audio from lullaby to white noise for the duration of his nap.

I capture the RTSP stream of a Nanit camera (great hardware, but no publicly accessible API) using Scrypted to authenticate and expose the stream on my local network. I then have BlueIris take a snapshot every 5 minutes to build a library of training data. The Scrypted RTSP stream is then consumed by OpenCV as you will see in the video processing library.

When the model determines my son is present or not present in the crib, it will toggle an input_boolean helper object using the Home Assistant REST API, this is demonstrated in toggle_josh_crib.py.

## Model Training
I used approximately 3000 images from sleep patterns over a 1 month time period. The file_sync.py will randomly select 20% of the total images and place them into a validation folder to validate the model as it trains to prevent over-fitting. 

Images were converted to greyscale during training to prevent the model from focusing on light changes when the curtains were open.

## Creds.py file
```bash
nas_host = "host_name"
nas_user = "user"
nas_password = "password" #only using rsync on linux
nas_path = "/share/training_data/josh_crib" #classified images are stored under here
model_name = 'crib_model.keras'
rtsp_url = "rtsp://1.2.3.4:567/a354jad"
home_assistant_url = "http://1.2.3.4:8123/api/services/input_boolean/"
ha_access_token = "ha long lived token"
ha_entity_id = "input_boolean.josh_in_crib"
```