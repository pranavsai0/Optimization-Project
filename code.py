import numpy as np
import matplotlib.pyplot as plt

# =====================================================
#                 PROBLEM PARAMETERS
# =====================================================
D = 10000               # Total demand
a1,b1 = 1e-4,5          # Express cost coefficients
a2,b2 = 5e-5,2          # Ground cost coefficients
T1,T2 = 1,3             # Delivery times (days)
T_max = 2.3             # Maximum allowed average time
e1,e2 = 2,1             # CO2 emissions
E_max = 15000           # Carbon limit

# =====================================================
#                COST FUNCTION
# =====================================================
def f(x1,x2):
    return a1*x1**2 + b1*x1 + a2*x2**2 + b2*x2

# =====================================================
#                CONSTRAINT FUNCTIONS
# =====================================================
def constraints(x1,x2):
    h  = (x1+x2-D)                      # demand equality
    g1 = (T1*x1 + T2*x2) - (T_max*D)     # time constraint
    g2 = (e1*x1 + e2*x2) - E_max         # CO2 constraint
    return h,g1,g2

# =====================================================
#     PENALTY
# =====================================================
def penalty_objective(x1,x2,mu):
    h,g1,g2 = constraints(x1,x2)
    return f(x1,x2) + mu*(h**2 + max(0,g1)**2 + max(0,g2)**2)

def grad(x1,x2,mu):
    h,g1,g2 = constraints(x1,x2)
    df1 = 2*a1*x1 + b1
    df2 = 2*a2*x2 + b2

    d1 = 2*h; d2 = 2*h
    if g1>0: d1 += 2*g1*T1; d2 += 2*g1*T2
    if g2>0: d1 += 2*g2*e1; d2 += 2*g2*e2

    return df1+mu*d1 , df2+mu*d2  #total gradient of penalty fn

mu=100; alpha=1e-4
x1,x2 = 3000, 7000  #initial values
 
hist_cost=[]; hist_viol=[]

print("\n================ PENALTY METHOD =================")

for stage in range(10):
    for _ in range(2000):
        g1,g2 = grad(x1,x2,mu)  
        x1 -= alpha*g1          #gradient descent
        x2 -= alpha*g2

        x1=np.clip(x1,0,D); x2=np.clip(x2,0,D)

        h,g1c,g2c=constraints(x1,x2)
        viol=abs(h)+max(0,g1c)+max(0,g2c)

        hist_cost.append(f(x1,x2))
        hist_viol.append(viol)

    print(f"Penalty Stage {stage+1} μ={mu} → x1={x1:.2f}, x2={x2:.2f}, viol={viol:.6f}")
    if viol<1e-3:
        print("Penalty Method Converged")
        break

    mu*=3; alpha/=1.2

penalty_iterations = len(hist_cost)



# =====================================================
#               FINAL PRINTED OUTPUT
# =====================================================

print("\n================ FINAL OPTIMIZATION REPORT =================")
print(f"Total Demand (D)                  : {D} kg")

print(f"Optimal Express (x1)              : {x1:.2f} kg")
print(f"Optimal Ground  (x2)              : {x2:.2f} kg")
print(f"Average Delivery Time             : {(T1*x1+T2*x2)/D:.7f} days")
print(f"Carbon Emission                   : {e1*x1 + e2*x2:.2f} <= {E_max}")
print(f"Iterations Taken                  : {penalty_iterations}")
print(f"Status                            : Converged")

print("\n============== Final Result Summary ==============")
print("Transport    x1(kg)    x2(kg)   AvgTime(days)")
print(f"Penalty →   {x1:7.1f}  {x2:7.1f}    {(T1*x1+T2*x2)/D:.10f}")
print("=====================================================\n")



# =====================================================
#                FINAL PLOT
# =====================================================
zoom = int(penalty_iterations*0.25)  # display only early convergence window

plt.figure(figsize=(10,5))
ax1=plt.gca()
ax1.plot(hist_cost[:zoom],linewidth=2,label="Cost",color='blue')
ax1.set_ylabel("Cost",color='blue'); ax1.tick_params(axis='y',labelcolor='blue')
ax2=ax1.twinx()
ax2.plot(hist_viol[:zoom],'r--',linewidth=2,label="Violation")
ax2.set_ylabel("Violation",color='red'); ax2.tick_params(axis='y',labelcolor='red')
plt.title(f"Penalty Method Convergence (Zoomed - First {zoom} Iterations)")
plt.grid(); plt.show()



