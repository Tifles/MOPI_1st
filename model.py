from diffusers import DiffusionPipeline
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# receive the request and translate

#Model_tr_Name = "Helsinki-NLP/opus-mt-ru-en" #online
Model_tr_Name = "./model_translate/ru-en-local"    #offline

tokenizer = AutoTokenizer.from_pretrained(Model_tr_Name)

model = AutoModelForSeq2SeqLM.from_pretrained(Model_tr_Name)

#tokenizer.save_pretrained('./model_translate/ru-en-local') #download model to catalog
#model.save_pretrained('./model_translate/ru-en-local')     #download model to catalog

#receive the request
req_name = input("Введите запрос:")

#translate for en
inputs = tokenizer(req_name, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=100)
out_text = tokenizer.batch_decode(output, skip_special_tokens=True)
req_name = out_text[0]

print(out_text)
print(req_name)

input("Нажмите Enter:")

model_paint_name = "stabilityai/stable-diffusion-xl-base-1.0" #online
#model_paint_name = "./model_paint/STBD10" #offline

pipe = DiffusionPipeline.from_pretrained(model_paint_name, torch_dtype=torch.float32, use_safetensors=True, variant="fp16", safety_checker=None)
#pipe.save_pretrained('./model_paint/STBD10')
#pipe.to("cuda")
pipe.enable_model_cpu_offload()

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()
#pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
#prompt = "An astronaut riding a green horse"

images = pipe(prompt=req_name).images[0]

images.show()
#images.save("../outputs/astronaut_rides_horse.png")
save_name=req_name.replace (' ','_')
images.save(save_name+".png")