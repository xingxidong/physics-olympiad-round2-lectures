"""生成书中计算实验的参考数据；全部数据由明确的模型计算，未调用外部 AI。"""
from pathlib import Path
import csv, json
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.special import ellipk
from scipy.optimize import minimize_scalar

ROOT=Path(__file__).resolve().parent
checks={}
# 非线性单摆：向下穿越平衡点的首事件是四分之一周期。
def period(angle_deg, tol=1e-11):
    a=np.deg2rad(angle_deg)
    def cross(t,y): return y[0]
    cross.direction=-1; cross.terminal=True
    s=solve_ivp(lambda t,y:[y[1],-np.sin(y[0])],(0,30),[a,0],events=cross,rtol=tol,atol=tol*0.01,max_step=.02)
    return 4*s.t_events[0][0]
periods=[]
for deg in [5,15,30,60,90,120]:
    a=np.deg2rad(deg); exact=4*ellipk(np.sin(a/2)**2)
    num=period(deg)
    periods.append({'angle_deg':deg,'T_over_T0':exact/(2*np.pi),'quadratic':1+a*a/16,'relative_error_numeric':abs(num/exact-1)})
checks['pendulum_max_relative_error']=max(v['relative_error_numeric'] for v in periods)
# 共振的三个峰和半功率点。
z=.1
resonance={'zeta':z,'displacement_peak':np.sqrt(1-2*z*z),'velocity_power_peak':1.,'half_power_low':np.sqrt(1+z*z)-z,'half_power_high':np.sqrt(1+z*z)+z}
# 失谐双振子：独立积分和解析谱分解比较。
coupling=[]
for delta in [0,.02,.06]:
    eps=.02; D=np.hypot(eps,delta)
    K=np.array([[1+eps+delta,-eps],[-eps,1+eps-delta]])
    w2,U=np.linalg.eigh(K); w=np.sqrt(w2); t=np.linspace(0,2*np.pi/(w[1]-w[0]),5001)
    exact=(U@((U.T@np.array([1.,0.]))[:,None]*np.cos(w[:,None]*t)))
    sol=solve_ivp(lambda t,y:np.r_[y[2:],-K@y[:2]],(0,t[-1]),[1.,0,0,0],t_eval=t,rtol=2e-11,atol=2e-13,max_step=.08)
    E=.5*np.sum(sol.y[2:]**2,axis=0)+.5*np.einsum('it,ij,jt->t',sol.y[:2],K,sol.y[:2])
    coupling.append({'epsilon':eps,'delta':delta,'omega_minus':w[0],'omega_plus':w[1],'first_envelope_peak_tau':np.pi/(w[1]-w[0]),'envelope_max_x2':eps/D,'leading_energy_fraction':eps**2/D**2,'max_position_error':float(np.max(abs(sol.y[:2]-exact))),'relative_energy_drift':float(np.max(abs(E/E[0]-1)))})
checks['coupled_max_position_error']=max(v['max_position_error'] for v in coupling)
# 有阻力抛体：无量纲，初速度为1，重力为1。
def projectile(angle_deg, beta=0., mode='quadratic', tol=1e-10):
    th=np.deg2rad(angle_deg)
    def rhs(t,y):
        v=np.hypot(y[2],y[3]); c=beta*v if mode=='quadratic' else beta
        return [y[2],y[3],-c*y[2],-1-c*y[3]]
    def ground(t,y): return y[1]
    ground.direction=-1;ground.terminal=True
    s=solve_ivp(rhs,(0,10),[0,0,np.cos(th),np.sin(th)],events=ground,rtol=tol,atol=tol*.01,max_step=.01)
    if not len(s.t_events[0]): raise RuntimeError('未找到落地事件')
    return float(s.y_events[0][0,0]),float(s.t_events[0][0])
projectiles=[]
for beta in [0,.2,1,3]:
    opt=minimize_scalar(lambda a:-projectile(a,beta)[0],bounds=(5,85),method='bounded',options={'xatol':1e-7})
    projectiles.append({'beta':beta,'optimal_angle_deg':float(opt.x),'max_range_scaled':-float(opt.fun),'range_45deg':projectile(45,beta)[0]})
checks['projectile_vacuum_error']=abs(projectile(45)[0]-1)
for angle in [20,45,70]:
    b=.3;r,tf=projectile(angle,b,'linear');th=np.deg2rad(angle)
    xa=np.cos(th)*(-np.expm1(-b*tf))/b
    ya=(np.sin(th)+1/b)*(-np.expm1(-b*tf))/b-tf/b
    checks[f'linear_drag_{angle}_error']=float(max(abs(r-xa),abs(ya)))
# 可复算的模拟测量数据。sigma 为设定的独立测量标准差。
x=np.arange(7)*.01; y=np.array([.20,1.01,1.83,2.70,3.65,4.71,5.87]);sigma=.02
with (ROOT/'弹簧测量.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['x_m','F_N','sigma_F_N']);w.writerows(zip(x,y,np.full(7,sigma)))
u=x/.06
A=np.c_[np.ones(7),u];B=np.c_[A,u**3]
fit1=np.linalg.lstsq(A,y,rcond=None)[0];fit3=np.linalg.lstsq(B,y,rcond=None)[0]
r1=y-A@fit1;r3=y-B@fit3
fit={'linear':{'b_N':fit1[0],'k_N_per_m':fit1[1]/.06,'residual_N':r1.tolist(),'chi_square':float(np.sum((r1/sigma)**2)),'dof':5},'cubic':{'b_N':fit3[0],'k_N_per_m':fit3[1]/.06,'c_N_per_m3':fit3[2]/.06**3,'residual_N':r3.tolist(),'chi_square':float(np.sum((r3/sigma)**2)),'dof':4}}
checks['linear_residual_orthogonality']=float(np.max(abs(A.T@r1)))
checks['cubic_residual_orthogonality']=float(np.max(abs(B.T@r3)))
# 曲轨周期与近似；用变换后的平滑积分避免转折点奇性。
bead=[]
for a in [.1,.3,.6]:
    ratio=2/np.pi*quad(lambda u:np.sqrt(1+a*a*np.sin(u)**2),0,np.pi/2,epsabs=1e-13)[0]
    bead.append({'a_over_R':a,'T_over_T0':ratio,'quadratic':1+a*a/4})
report={'pendulum':periods,'resonance':resonance,'coupling':coupling,'projectile':projectiles,'spring_fit':fit,'bead':bead,'checks':checks}
assert checks['pendulum_max_relative_error']<1e-8
assert checks['coupled_max_position_error']<1e-7
assert max(v for k,v in checks.items() if k.startswith('linear_drag'))<1e-8
assert checks['projectile_vacuum_error']<1e-8
report['status']='passed'
(ROOT/'参考结果.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
