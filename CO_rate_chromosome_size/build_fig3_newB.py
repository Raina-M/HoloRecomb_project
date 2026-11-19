"""Replace panel b of Fig_3.pdf with the revised panel (vector overlay)."""
import pymupdf, sys
src=sys.argv[1]; d=pymupdf.open(src); p=d[0]
p.draw_rect(pymupdf.Rect(48,165,219.5,330),color=None,fill=(1,1,1),overlay=True)
pan=pymupdf.open('Fig3b_CO_rate_vs_chrsize_white.pdf')
p.show_pdf_page(pymupdf.Rect(49,166,49+169,166+161),pan,0)
d.save('Fig_3_revised.pdf',garbage=3,deflate=True)
pymupdf.open('Fig_3_revised.pdf')[0].get_pixmap(dpi=300).save('Fig_3_revised_preview.png')
