"""
Práctica 3: Sistema musculoesquelético

Departamento de Ingeniería Eléctrica y Electrónica, Ingeniería Biomédica
Tecnológico Nacional de México [TecNM - Tijuana]
Blvd. Alberto Limón Padilla s/n, C.P. 22454, Tijuana, B.C., México

Nombre del alumno: Rafael Herrera Aguilar
Número de Control: 22212258
Correo institucional: l22212258@tectijuana.edu.mx

Asignatura: Modelado de Sistemas Fisiológicos
Docente: Dr. Paul Antonio Valle Trujillo; paul.valle@tectijuana.edu.mx
"""

#Librerías para cálculo numérico y generación de gráficas
import numpy as np
import math as m
import matplotlib.pyplot as plt
import control as ctrl
from scipy import signal
import pandas as pd

u = np.array(pd.read_excel('signal.xlsx',header=None))

#Datos de la simulación
x0,t0,tF,dt,w,h = 0,0,10,1E-3,10,5
N = round((tF-t0)/dt) + 1
t = np.linspace(t0,tF,N)
u = np.reshape(signal.resample(u, len(t)),-1)
Fs0 = np.zeros(N); Fs0[round(1/dt):round(2/dt)] = 1

def cardio(Cs,Cp,R):
    num = [Cs*R,1-0.25]
    den = [R*(Cp+Cs),1]
    sys = ctrl.tf(num,den)
    return sys

#Función de transferencia: Individuo sano (control)
Cs,Cp,R = 10E-6, 100E-6, 100
syssano = cardio(Cs,Cp,R)
print(f"Individuo sano (control): {syssano}")

#Función de transferencia: Paciente (caso)
Cs,Cp,R = 10E-6, 100E-6, 10E3
syspaciente = cardio(Cs,Cp,R)
print(f"Individuo sano (control): {syspaciente}")

#Respuestas en lazo abierto
_,Fs1 = ctrl.forced_response(syssano,t,Fs0,x0)
_,Fs2 = ctrl.forced_response(syspaciente,t,Fs0,x0)

clr1 = np.array([230, 39, 39])/255
clr2 = np.array([0, 0, 0])/255
clr3 = np.array([67, 0, 255])/255
clr4 = np.array([22, 97, 14])/255
clr5 = np.array([250, 129, 47])/255
clr6 = np.array([145, 18, 188])/255

fg1 = plt.figure()
plt.plot(t,Fs0,'-',linewidth=1,color=clr1,label='F(t)')
plt.plot(t,Fs1,'-',linewidth=1,color=clr2,label='Fs(t):Control')
plt.plot(t,Fs2,'-',linewidth=1,color=clr3,label='Fs(t):Caso')
plt.grid(False)
plt.xlim(0,10); plt.xticks(np.arange(0,11,1))
plt.ylim(-0.1,1.1); plt.yticks(np.arange(0,1.2,0.2))
plt.xlabel('t[s]')
plt.ylabel('Pp(t) [V]')
plt.legend(bbox_to_anchor=(0.5,-0.2),loc='center',ncol=3)
plt.show()
fg1.set_size_inches(w,h)
fg1.tight_layout()
fg1.savefig('Sistema musculoesquelético python.png',dpi=600,bbox_inches='tight')
fg1.savefig('Sistema musculoesquelético python.pdf',bbox_inches='tight')

def controlador(kP,kI):
    Cr = 1E-6
    Re = 1/(kI*Cr)
    Rr = kP*Re
    numPI = [Rr*Cr,1]
    denPI = [Re*Cr,0]
    PI = ctrl.tf(numPI,denPI)
    return PI

PI = controlador(0.0219712691125975,41303.6112882169)
X = ctrl.series(PI,syspaciente)
tratamiento_PI = ctrl.feedback(X,1,sign=-1)

_,Fs3 = ctrl.forced_response(tratamiento_PI,t,Fs1,x0)

fg2 = plt.figure()
plt.plot(t,Fs0,'-',linewidth=1,color=clr1,label='F(t)')
plt.plot(t,Fs1,'-',linewidth=1,color=clr2,label='Fs(t): Control')
plt.plot(t,Fs2,'-',linewidth=1,color=clr3,label='Fs(t): Caso')
plt.plot(t,Fs3,':',linewidth=2,color=clr4,label='Fs(t): Tratamiento')
plt.grid(False)
plt.xlim(0,10); plt.xticks(np.arange(0,11,1))
plt.ylim(-0.1,1.1); plt.yticks(np.arange(0,1.2,0.2))
plt.xlabel('t[s]')
plt.ylabel('Fi(t) [V]')
plt.legend(bbox_to_anchor=(0.5,-0.2),loc='center',ncol=5)
plt.show()
fg2.set_size_inches(w,h)
fg2.tight_layout()
fg2.savefig('Sistema musculoesquelético python PI.png',dpi=600,bbox_inches='tight')
fg2.savefig('Sistema musculoesquelético python PI.pdf',bbox_inches='tight')

