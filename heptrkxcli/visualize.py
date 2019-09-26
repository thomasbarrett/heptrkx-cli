from PIL import Image, ImageDraw, ImageFont
from heptrkxcli.config import config_file
import numpy as np
import os

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

def visualize_hitgraph(folder, name, hit_graph):    
    img = Image.new('RGBA', (1000,1000), (255,255,255,255))
    draw = ImageDraw.Draw(img)

    for (x,y,z) in hit_graph['nodes']:
        (x, y) = (1000 * x, 1000 * y + 500)
        draw.ellipse([x-1,y-1,x+1,y+1], fill='#000000')

    for (u, v, t) in zip(hit_graph['senders'], hit_graph['receivers'], hit_graph['edges']):
        (x1, y1, z1) = hit_graph['nodes'][u]
        (x2, y2, z2) = hit_graph['nodes'][v]
        (x1, y1) = (1000 * x1, 1000 * y1 + 500)
        (x2, y2) = (1000 * x2, 1000 * y2 + 500)
        if (t > 0.04):
            draw.line([(x1, y1), (x2, y2)], fill=(255,255,255,0))
    
    os.makedirs(folder, exist_ok=True)
    img.save(folder + '/track%d.png' % name, "PNG")


if __name__ == '__main__':
    export_track()