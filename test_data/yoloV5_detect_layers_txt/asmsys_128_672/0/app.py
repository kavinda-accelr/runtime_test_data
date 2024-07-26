from os import path

dets_path = 'asmsys_results'
dets_outs_path = ''
image_path = 'asmsys_results'
image_out_path = ''

dets_tensors = ['yolov5-best-ext_0.txt', 'yolov5-best-ext_1.txt', 'yolov5-best-ext.txt']
dets_tensors_outs = ['o1.txt', 'o2.txt', 'o3.txt']
image_txt = 'image.txt'
image_txt_out = 'img.txt'

det_dims = ['(1, 16, 84, 75)', '(1, 8, 42, 75)', '(1, 4, 21, 75)']
image_dims = 'Layer 0 i (1, 128, 672, 3)'

# output_mult = [94, 64, 64]
# output_shift = [6, 6, 6]
output_mult = [1, 1, 1]
output_shift = [1, 1, 1]

output_mult_img = 0
output_shift_img = 255

def inverse_quantization_dets(val, output_mult, output_shift):
    return (val*output_shift)/output_mult
    
def inverse_quantization_image(val, output_mult, output_shift):
    return (val*output_shift)/output_mult

for n in range(len(dets_tensors)):
    with open(path.join(dets_path, dets_tensors[n]), 'r') as f:
        data = f.read().split()[10:]

    with open(path.join(dets_outs_path, dets_tensors_outs[n]), 'w') as f:
        f.write(f'{det_dims[n]}\n')
        for num in data:
            f.write(f'{inverse_quantization_dets(int(num), output_mult[n], output_shift[n])}\n')

# image
with open(path.join(image_path, image_txt), 'r') as f:
    data = f.read().split()[2:]

int_data = []
for num in data:
    int_data.append(int(num))

output_mult_img = max(int_data)
print(output_mult_img)
with open(path.join(image_out_path, image_txt_out), 'w') as f:
    f.write(f'{image_dims}\n')
    for num in int_data:
        f.write(f'{int(inverse_quantization_image(num, output_mult_img, output_shift_img))}\n') 

