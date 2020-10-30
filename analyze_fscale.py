# -*- coding: utf-8 -*- 

import numpy as np
import sys
import pandas
import matplotlib.pyplot as plt

def load_f(name):
	with open(name,'r') as f:
		data=f.read()
	return data

z_re = sys.argv[1]
suffix = sys.argv[2]
path='/fs/scratch/PCON0003/osu10670/fmass/HMZ'+z_re+suffix
path_nv='/fs/scratch/PCON0003/osu10670/fmass/HMZ'+z_re+'-nv'+suffix


class f_scale:

	def __init__(self,file_name,a):
		parsec=3.085677581e16 # m per parsec
		self.file_name=file_name	
		self.H_0=67.74e3/(parsec*10**6) # in units of s^-1, Hubble constants now, 67.74 km/s/mpc
		self.Omega_m=0.3089 # Omega_m = 0.3089+-0.0062
		self.G=6.674e-11  #6.674×10−11 m3*kg−1*s−2 ### 4.30091(25)×10−3 pc*M_solar-1*(km/s)^2
		self.solar_m= 1.98847e30 #(1.98847±0.00007)×10^30 kg

		self.a_dec=0.000942791416826941
		self.a_t=a # final scale factor

		self.t_dec=2*self.a_dec**(3./2)/3/self.H_0/np.sqrt(self.Omega_m)
		self.t= 2*self.a_t**(3./2)/3/self.H_0/np.sqrt(self.Omega_m)

		self.psi_dec=self.psi(self.t_dec)
		# print('psi_dec='+str(self.psi_dec)+'   '+str((self.t_dec/self.t)**(2./3)))

		self.rho_mo=3*self.Omega_m*self.H_0**2/8/np.pi/self.G 

		# print(self.H_0,self.t_dec,self.t)

	def cs_extract(self):
		k=1.38069e-23 # Boltzmann constant @ J*K-1
		m_p=1.67262192369e-27 # proton mass @ kg

		data=load_f(self.file_name)
		data=[float(i) for i in data.split()]
		data=np.reshape(np.array(data),(int(len(data)/9),9))

		T_0=np.mean(data[:,4][np.abs(data[:,3]-1)<0.01])
		logT_0=np.mean(np.log10(data[:,4][np.abs(data[:,3]-1)<0.01]))
		rho_mean = np.mean(data[:,3])
		gamma=np.array([((np.log10(i[4])-np.log10(T_0))/np.log10(i[3])+1) for i in data if (i[3]-1>0.05 and i[3]<300)])
		gamma=np.mean(gamma[:][gamma>-1]) 

		mu=m_p/2 # reduced mass @ pure H plasma
		c_s = np.sqrt(gamma*k*T_0/mu) # calculate sound speed

		return c_s, gamma, T_0,logT_0

	def kernel(self,psi):
		return 2*(1-3*self.psi_dec/psi+2*(self.psi_dec/psi)**(3./2))*(1-psi**(1./2))/(1-3*self.psi_dec+2*self.psi_dec**(3./2))

	def psi(self,time):
		return self.a(time)/self.a_t

	def a(self,time):
		return (3*self.H_0*np.sqrt(self.Omega_m)/2)**(2./3)*time**(2./3)

	def a_deriv(self,time):
		return 2./3.*(3*np.sqrt(self.Omega_m)*self.H_0/2)**(2./3)*time**(-1./3)

	def results(self):
		psi=self.psi_dec
		t=self.t_dec

		ker =[]
		p=[]
		cs_aH = []
		a=[]
		a_dot=[]
		ts=[]

		kF_inverse2=0
		dpsi=(1-self.psi_dec)/1000.
		c_s,gamma,T_0,logT_0=self.cs_extract()
		for i in range(1000):
			ker.append(self.kernel(psi))
			p.append(psi)
			cs_aH.append(c_s**2/(self.a_deriv(t))**2)
			a.append(self.a(t))
			a_dot.append(self.a_deriv(t))
			ts.append(t)

			dt=3./2*np.sqrt(psi)*self.t*dpsi
			kF_inverse2+=self.kernel(psi)*c_s**2/(self.a_deriv(t))**2*dpsi
			psi+=dpsi
			t+=dt

		print(cs_aH)
		print(a)
		print(a_dot)
		print(ts)
		print(ker)
		print(p)

		# plt.plot(np.array(p),np.array(ker),label=r'$\psi_{dec}=$'+str(self.psi_dec))
		# plt.show()

		# plt.plot(np.array(p),np.array(cs_aH),label=r'$c_s/(aH)^2$')
		# plt.show()
		print(np.max(cs_aH))
		kF=np.sqrt(1/kF_inverse2)
		MF=self.rho_mo*4./3*np.pi**4*(kF_inverse2)**(3./2)/self.solar_m*67.74/100 # in units of M_solar/h

		return MF,kF,kF_inverse2,c_s,gamma,T_0,logT_0

for fold in [path_nv,path]:
	if fold==path:
		print('Simulation results with streaming velocity:')
	else:
		print('Simulation results without streaming velocity:')
	a=load_f(fold+'/outputs.txt').split('\n')
	a.pop(-1)
	a=[float(i) for i in a]
	z=[1/i-1 for i in a]
	MF=[]
	kF=[]
	kF_inverse2=[]
	c_s=[]
	gamma=[]
	T_0=[]
	logT_0=[]

	for i in range(5):
		f=f_scale(fold+'/snapshot_00'+str(i)+'.extract',a[i])
		MF_tmp,kF_tmp,kF_inverse2_tmp,c_s_tmp,gamma_tmp,T_0_tmp,logT_0_tmp = f.results()
		MF.append(MF_tmp)
		kF.append(kF_tmp)
		kF_inverse2.append(kF_inverse2_tmp)
		c_s.append(c_s_tmp)
		gamma.append(gamma_tmp)
		T_0.append(T_0_tmp)
		logT_0.append(logT_0_tmp)

		print('At redshift z='+str(z[i])+', a='+str(a[i]))
		print('Filtering mass M_F='+str(MF[i])+' , filtering scale k_F='+str(kF[i])+', k_F^-2='+str(kF_inverse2[i]))
		print('Sound speed c_s='+str(c_s[i])+', gamma='+str(gamma[i]))
		print('Temperature at mean density T_0='+str(T_0[i])+', log10T_0='+str(logT_0[i])+'\n')
	data={'a':a,'z':z,'MF':MF,'kF':kF,'c_s':c_s,'gamma':gamma,'T_0':T_0,'logT_0':logT_0,'kF_inverse2':kF_inverse2}
	filename='f_scale_'+z_re+suffix+'_test.csv' if fold==path else 'f_scale_'+z_re+'-nv'+suffix+'_test.csv'
	print('data --> '+ filename+'\n')
	pandas.DataFrame(data).to_csv(filename)

# data=[float(i) for i in data.split()]
# data=np.reshape(data,(len(data)/9,9))

# T_0=np.mean(data[:,4][[np.abs(data[:,3]-1)<0.01]]) # temperature at mean density
# #7689.7784237517035 with sv
# #7699.7760521545006 without sv

# #np.mean(np.log10(data[:,4][[np.abs(data[:,3]-1)<0.01]]))
# #3.8811225512246916 with sv
# #3.8816260318191143 without sv

# gamma=[((np.log10(i[4])-np.log10(T_0))/np.log(i[3])+1) for i in data]
# gamma=np.mean(gamma[gamma>-1]) 
# # 1.2059719361780816 with sv
# # 1.1725213083967612 without sv
# mu=m_p/2 # reduced mass @ pure H plasma
# c_s = np.sqrt(gamma*k*T_0/mu)

#TimeBegin           a=0.000942791416826941






