"""Reproduce the event, sensitivity, and sampling calculations in lecture 4."""
from pathlib import Path
import math,random,csv
OUT=Path(__file__).with_name('事件与抽样实验数据')
def bisect(f,a,b,tol=1e-13):
    fa=f(a);fb=f(b)
    if fa*fb>=0:raise ValueError('The interval must bracket a non-endpoint root.')
    while b-a>tol:
        m=(a+b)/2;fm=f(m)
        if fa*fm<=0:b=m
        else:a=m;fa=fm
    return (a+b)/2
def save(name,header,rows):
    OUT.mkdir(exist_ok=True)
    with (OUT/name).open('w',encoding='utf-8-sig',newline='')as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)
def mean_se(values):
    n=len(values);mean=math.fsum(values)/n
    var=math.fsum((x-mean)**2 for x in values)/(n-1)
    return mean,math.sqrt(var/n)
def events():
    y=lambda t:87.5*(-math.expm1(-.4*t))-25*t
    root=bisect(y,1,3);rows=[]
    for h in [.2,.1,.05,.025]:
        a=math.floor(root/h)*h;b=a+h
        t=a+h*y(a)/(y(a)-y(b));rows.append([h,t,t-root,a,b])
    save('事件插值.csv',['step_s','linear_event_s','bias_s','left_s','right_s'],rows)
    th0=math.asin(5/6)
    vel=lambda th:.6*(math.cos(th0)-math.cos(th))-.3*(th-th0)
    stop=bisect(vel,math.pi,4.1)
    save('摩擦切换.csv',['event','phase_rad'],[['start',th0],['stop',stop],['reverse_start',math.pi+th0]])
def trajectory(theta):
    a=10*math.sin(theta);b=10*math.cos(theta);k=.4;g=10
    y=lambda t:(a+g/k)*(-math.expm1(-k*t))/k-g*t/k
    T=bisect(y,1e-8,3);e=math.exp(-k*T)
    Tp=-(b*(1-e)/k)/((a+g/k)*e-g/k)
    return T,b*(1-e)/k,-a*(1-e)/k+b*e*Tp
def sensitivities():
    angle=bisect(lambda th:trajectory(th)[2],.2,1.2)
    rows=[]
    for degrees in [30,35,40,math.degrees(angle),45,50,55,60]:
        T,R,dR=trajectory(math.radians(degrees));rows.append([degrees,T,R,dR])
    save('最佳角与灵敏度.csv',['angle_deg','flight_s','range_m','range_derivative_m_per_rad'],rows)
def sampling():
    rows=[];rng=random.Random(20261027);N=200000
    direct=[float(-math.log1p(-rng.random())>8)for _ in range(N)]
    estimate,se=mean_se(direct);rows.append(['direct',N,estimate,se,math.exp(-8)])
    lam=1/8;N=1060;rng=random.Random(20261028)
    values=[]
    for _ in range(N):
        x=-math.log1p(-rng.random())/lam
        values.append(math.exp(-(1-lam)*x)/lam if x>8 else 0.)
    estimate,se=mean_se(values);rows.append(['importance',N,estimate,se,math.exp(-8)])
    save('高能尾部抽样.csv',['method','samples','estimate','estimated_standard_error','reference'],rows)
if __name__=='__main__':
    events();sensitivities();sampling()
    print(OUT)
