## Picture folder
```
origin_image_dir = "./sample/origin/" # Original file
train_image_dir = "./sample/train/" # training set
test_image_dir = "./sample/test/" # test set
api_image_dir = "./sample/api/" # Storage path of the image received by api
online_image_dir = "./sample/online/" # The storage path of the image obtained from the verification code url
```
## Model folder
```
model_save_dir = "./model/" # The storage path of the trained model
```
## Picture related parameters
```
image_width = 80 # Image width
image_height = 40 # Picture height
max_captcha = 4 # Number of characters in the verification code
image_suffix = "jpg" # Picture file suffix
```
## Whether to import tags from the file
```
use_labels_json_file = False
```
## Verification code character related parameters
```
char_set = "0123456789abcdefghijklmnopqrstuvwxyz"
char_set = "abcdefghijklmnopqrstuvwxyz"
char_set = "0123456789"
```
## Online identification of remote verification code address
```
remote_url = "http://127.0.0.1:6100/captcha/"
```
## Training related parameters
```
cycle_stop = 3000 # Stop after the specified number of iterations
acc_stop = 0.99 # Stop after the specified accuracy rate
cycle_save = 500 # Save once for each specified number of training rounds (overwrite the previous model)
enable_gpu = 0 # Use GPU or CPU, you need to install the corresponding version of tensorflow-gpu==1.7.0 to use GPU