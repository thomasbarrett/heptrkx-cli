from PIL import Image, ImageDraw, ImageFont
from heptrkxcli.config import config_file
import numpy as np

def transform_coordinates(x, y):
    return (x+1000,y+1000)

def export_track(name, data):
    print('drawing track')
    
    if name == 0:
        return

    img = Image.new('RGBA', (2000,2000), (255,255,255,255))
    draw = ImageDraw.Draw(img)

    for (x,y,z,cx,cy,r) in data.to_numpy():
        (x, y) = transform_coordinates(x, y)
        (cx, cy) = transform_coordinates(cx, cy)
        draw.ellipse([x-5,y-5,x+5,y+5], fill='#000000')
        draw.ellipse([cx-r,cy-r,cx+r,cy+r], outline='#888888', width=2)

    print('saving track')

    img.save('figures/track%d.png' % name, "PNG")

if __name__ == '__main__':
    export_track()