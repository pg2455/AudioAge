from shell_utils import generate_shell_script, FixedDict
BASE_ID = "GAUSSIAN"

base_params = {
  "timeframe":20,
  "windowfn":"gaussian",
  "modelarch":"resnet18",
  "lr":0.0001,
  "batchsize":64,
  "xbatches":100,
  "schedulerfactor":0.5,
  "schedulerpatience":50,
  "l2":0.0,
  "epoch":100,
  "job_name":"base",
  "lastupdate":10
}

params = FixedDict(base_params)
ID = BASE_ID + "res18"
generate_shell_script(params, ID, CUDA=False)
