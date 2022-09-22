from PIL import Image
x=11
#Create an Image Object from an Image
im = Image.open('Webp.net-resizeimage.png')
# rgb_im = im.convert('RGB')
im.crop((0,100 , 40, 580-0)).save(f'0.png', quality=100)
