"""Render saved statistical results with existing ReportLab; no market/model input.

Run with the bundled document Python. Keeps the experiment environment unchanged.
"""
import json
import math
from pathlib import Path
from reportlab.graphics.shapes import Drawing,String,Polygon,Rect,Line
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics import renderSVG
from reportlab.lib import colors

HERE=Path(__file__).resolve().parent

def plot():
    result=json.loads((HERE/'result_3_5_7.json').read_text())['results']['analysis_5_decay']
    groups=result['groups'];maximum=1.2
    for tiers in groups.values():
        for g in tiers.values():
            if g['median_curve']: maximum=max(maximum,max(g['median_curve']))
            if g['pointwise_ci95']: maximum=max(maximum,max(q[1] for q in g['pointwise_ci95']))
    ymax=math.ceil(maximum*2)/2
    d=Drawing(980,650);d.add(Rect(0,0,980,650,fillColor=colors.white,strokeColor=None))
    def text(x,y,s,size=12,color='#172933',anchor='start',font='Times-Roman'):
        d.add(String(x,y,s,fontName=font,fontSize=size,fillColor=colors.HexColor(color),textAnchor=anchor))
    text(490,618,'Observed volatility after a shock',23,anchor='middle',font='Times-Bold')
    text(490,597,'Frozen January / February events | 10-minute rolling volatility / pre-shock baseline',12,anchor='middle')
    d.add(Line(180,577,210,577,strokeColor=colors.HexColor('#185c76'),strokeWidth=2))
    text(217,573,'Median across events',11)
    d.add(Rect(370,572,24,10,fillColor=colors.HexColor('#dcebf0'),strokeColor=None))
    text(401,573,'Pointwise 95% day-block interval',11)
    d.add(Line(629,577,657,577,strokeColor=colors.HexColor('#986e30'),strokeWidth=1.2,strokeDashArray=[4,3]))
    text(664,573,'Recovery threshold = 1.2',11)
    for row,tier in enumerate(['major','small']):
        for col,month in enumerate(['2026-01','2026-02','combined']):
            g=groups[month][tier];r=g['recovery']
            x=60+col*320;y=331-row*261;w=267;h=166
            title=('January' if month=='2026-01' else 'February' if month=='2026-02' else 'January + February')+' | '+('Major assets' if tier=='major' else 'Small assets')
            text(x,y+h+43,title,14,font='Times-Bold')
            recovery='>120' if r['median_confirmation_minutes'] is None else str(r['median_confirmation_minutes'])
            text(x,y+h+26,f"Curve confirmation = {g['median_curve_confirmation_minutes']} min | event median = {recovery} min",10)
            text(x,y+h+11,f"n={g['n']}/{g['n_expected']}; unconfirmed individual events: {r['n_right_censored_gt_120']}/{g['n']}",10)
            band=g['pointwise_ci95'];curve=g['median_curve']
            if band:
                points=[]
                for k,q in enumerate(band):points.extend([x+w*k/120,y+h*q[0]/ymax])
                for k in range(120,-1,-1):points.extend([x+w*k/120,y+h*band[k][1]/ymax])
                d.add(Polygon(points,fillColor=colors.HexColor('#dcebf0'),strokeColor=None))
            lp=LinePlot();lp.x=x;lp.y=y;lp.width=w;lp.height=h
            lp.data=[[(i,v) for i,v in enumerate(curve or [0]*121)],[(0,1.2),(120,1.2)]]
            lp.lines[0].strokeColor=colors.HexColor('#185c76');lp.lines[0].strokeWidth=1.7
            lp.lines[1].strokeColor=colors.HexColor('#986e30');lp.lines[1].strokeWidth=1.1;lp.lines[1].strokeDashArray=[4,3]
            lp.xValueAxis.valueMin=0;lp.xValueAxis.valueMax=120;lp.xValueAxis.valueStep=30
            lp.yValueAxis.valueMin=0;lp.yValueAxis.valueMax=ymax;lp.yValueAxis.valueStep=.5 if ymax<=3 else 1
            for axis in [lp.xValueAxis,lp.yValueAxis]:
                axis.labels.fontName='Times-Roman';axis.labels.fontSize=10;axis.strokeColor=colors.HexColor('#63747e');axis.strokeWidth=.5
            lp.yValueAxis.visibleGrid=True;lp.yValueAxis.gridStrokeColor=colors.HexColor('#ccd6da');lp.yValueAxis.gridStrokeWidth=.25
            d.add(lp)
            text(x+w/2,y-31,'Minutes after shock t0',11,anchor='middle')
    text(490,23,'Recovery requires 10 consecutive points <= 1.2; later shocks retained. Historical description, not a forecast guarantee.',11,anchor='middle')
    target=HERE/'decay_curves.svg'
    if target.exists() and not __import__("sys").argv.count("--refresh-render"):raise FileExistsError(target)
    svg=renderSVG.drawToString(d)
    svg=svg.replace("font-family: Times-Bold", "font-family: Times New Roman; font-weight: bold").replace("font-family: Times-Roman", "font-family: Times New Roman")
    with target.open('w' if target.exists() else 'x') as f:f.write(svg)
    print(target)

if __name__=='__main__':plot()
