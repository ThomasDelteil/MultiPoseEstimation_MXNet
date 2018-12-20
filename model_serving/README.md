## Create the model archive 

Note that you need the `.params` file for it to work in the next step. Pre-trained files will be available shortly.

`model-archiver --model-name pose_V0.1 --model-path model_serving/files --handler pose_service:pose_inference --export-path 
model_serving/`

## Serve the model
`mxnet-model-server --start --models pose=pose_V0.1.mar --model-store=model_serving --mms-config model_serving/config.properties`
