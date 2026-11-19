import re, copy
from lxml import etree
NS='{http://www.w3.org/2000/svg}'
P=etree.XMLParser(huge_tree=True)
IMG=[(454.4,127.2,502.3,175.2),(454.5,69.8,502.6,117.9),(454.4,185.9,502.2,234.0),(454.0,243.0,502.0,291.1),(454.6,7.8,502.8,56.0),
     (158.3,432.3,233.3,505.8),(80.9,353.7,156.2,429.0),(2.7,431.2,78.1,506.5),(3.4,353.7,78.7,429.0),(80.6,431.2,155.9,506.5),(158.7,355.1,232.2,428.7)]
B_SHIFT={'R. breviuscula':1.4,'R. nervosa':3.8,'R. ciliata':4.4,'R. colorata':6.4}
def text_of(el): return ''.join(el.itertext())
def common(root):
    for el in root.iter():
        if not isinstance(el.tag,str): continue
        ff=el.get('font-family')
        if ff and '+' in ff:
            base=ff.split('+',1)[1]; el.set('font-family','Arial')
            if 'Bold' in base: el.set('font-weight','bold')
            if 'Italic' in base: el.set('font-style','italic')
        if el.tag==NS+'text' and ''.join(el.itertext())=='R. holochoenoides':
            ts=el.findall(NS+'tspan'); f=ts[0]; f.text='R. holoschoenoides'; f.set('x',f.get('x').split()[0])
            for t in ts[1:]: el.remove(t)
def whiten(root):
    def in_op(el):
        p=el.getparent()
        while p is not None:
            if p.get('opacity')=='.501999': return True
            p=p.getparent()
        return False
    for el in root.iter():
        if not isinstance(el.tag,str): continue
        tag=el.tag.replace(NS,'')
        if tag=='path' and el.get('d')=='M0 510.236H510.236V0H0Z' and el.get('fill') is None: el.set('fill','#ffffff')
        if el.get('stroke')=='#ffffff': el.set('stroke','#000000')
        if tag=='path' and el.get('fill')=='#ffffff' and in_op(el): el.set('fill','#7a7a7a')
        if tag=='path' and el.get('fill')=='#00ff00': el.set('fill','#169c3c')
        if tag=='text' and el.get('fill')=='#ffffff':
            ts=el.find(NS+'tspan'); 
            if ts is None: el.set('fill','#000000'); continue
            # multiple tspans: decide per tspan
            for ts in el.findall(NS+'tspan'):
                x=float(ts.get('x').split()[0]); y=510.236+float(ts.get('y').split()[0])
                inside=any(a-1<=x<=c and b<=y<=d+1 for a,b,c,d in IMG)
                ts.set('fill','#ffffff' if inside else '#000000')
                t=(ts.text or '').strip()
                if x>440 and t in B_SHIFT:
                    ts.set('fill','#000000'); ts.set('y',str(float(ts.get('y'))-B_SHIFT[t]))
            el.set('fill','#000000')
def build(theme):
    base=etree.parse('orig.svg',P).getroot(); common(base)
    if theme=='white': whiten(base)
    ef=etree.parse(f'ef_{theme}.svg',P).getroot()
    W=510.236; H=510.236
    out=etree.Element(NS+'svg',nsmap={None:'http://www.w3.org/2000/svg','xlink':'http://www.w3.org/1999/xlink'})
    out.set('version','1.1'); out.set('width',f'{W}pt'); out.set('height',f'{H}pt'); out.set('viewBox',f'0 0 {W} {H}')
    bg=etree.SubElement(out,NS+'rect',x='0',y='0',width=str(W),height=str(H),fill='#000000' if theme=='black' else '#ffffff')
    base.set('x','0'); base.set('y','0'); base.set('width',str(W)); base.set('height','510.236')
    out.append(base)
    bgc='#000000' if theme=='black' else '#ffffff'
    etree.SubElement(out,NS+'rect',x='237',y='348.5',width=str(W-237),height=str(H-348.5),fill=bgc)
    etree.SubElement(out,NS+'rect',x='237',y='343',width='14',height='6',fill=bgc)
    nd=etree.parse(f'Fig2d_normalised_NND_{theme}.svg',P).getroot()
    nd.set('x','237'); nd.set('y','344'); nd.set('width','273.2'); nd.set('height','166.2'); nd.set('viewBox','0 0 273.2 166.2')
    out.append(nd)
    cb={3:(145.15,171.85),9:(175.7,196.76),4:(200.6,224.96),6:(228.81,251.67),7:(255.52,277.85),8:(281.7,303.14),5:(306.99,329.55),2:(333.39,362.92),1:(366.78,401.42)}
    acc='#ff3dc8' if theme=='black' else '#d4148f'
    for c,e in [(2,'s'),(5,'s'),(3,'e'),(4,'e'),(6,'s'),(7,'e'),(8,'s'),(9,'s')]:
        x=cb[c][0] if e=='s' else cb[c][1]; y0=3.29
        etree.SubElement(out,NS+'polygon',points=f'{x-1.7},{y0-2.9} {x+1.7},{y0-2.9} {x},{y0-0.2}',fill=acc)
    etree.ElementTree(out).write(f'Fig_2_revised_newD_{theme}.svg',xml_declaration=True,encoding='utf-8')
    import cairosvg
    cairosvg.svg2pdf(url=f'Fig_2_revised_newD_{theme}.svg',write_to=f'Fig_2_revised_newD_{theme}.pdf')
    cairosvg.svg2png(url=f'Fig_2_revised_newD_{theme}.svg',write_to=f'Fig_2_revised_newD_{theme}_preview.png',output_width=1800)
build('white'); build('black')
