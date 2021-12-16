# import sys, platform, os
# import matplotlib
# from matplotlib import pyplot as plt
# import numpy as np
# import camb
# from camb import model, initialpower

# pars = camb.CAMBparams()
# pars.set_cosmology(H0=67.74, ombh2=0.022, omch2=0.119,zrei=8.0)
# pars.InitPower.set_params(As=2.1e-9,ns=0.958)
# pars.set_matter_power(redshifts=[4], kmax=819.14595) # k in unit of Mpc^(-1)
# results = camb.get_results(pars)
# kh, z, pk = results.get_matter_power_spectrum(minkh=1.14102e-5, maxkh=1209.25, npoints = 636)
# s8 = np.array(results.get_sigma8())
# # for k=1.14102e-05 to 1209.25 h/Mpc, kmin=7.7e-4,kmax=81914.595

from scipy import interpolate
from scipy import integrate
from scipy.misc import derivative

import numpy as np
#from cluster_toolkit import massfunction
import matplotlib.pyplot as plt
import re, os, sys
import pandas as pd
from matplotlib import cm

# Constants
# rho_crit=2.775808e11 # units are M_solar h^2/Mpc^3
parsec=3.085677581e16 # m per parsec
H_0=67.74 # Hubble constants now, 67.74 km/s/mpc
G=4.30091e-9  # 6.674×10−11 m3*kg−1*s−2 ### 4.30091(25)×10−3 Mpc*M_solar-1*(km/s)^2
solar_m= 1.98847e30 # (1.98847±0.00007)×10^30 kg

rho_crit=3*H_0**2/8/np.pi/G*H_0/100 # M_solar/(Mpc)^3/h
# z=5.5

Omega_m=0.3089 # Omega_m = 0.3089+-0.0062
rhom=rho_crit*Omega_m #*(1+z)**3
f_b=0.17 # baryon fraction
#MF=3039380.57457326 # in unit of M_solar/h. with velocity
# MF=[2917922.227,2896770.297] # no streaming velocity, streaming velocity

A=0.186
a=1.47
b=2.57
c=1.19

Mhalos=np.logspace(7,16,10000)
ks=np.logspace(-4,3,1000)

sigmas=[]

############## Calculation of sigma_8 ####################
# data=np.loadtxt('zobs_5.500_pk.dat')

# k=data[:,0]
# pk=data[:,1]
# pk_fun=interpolate.interp1d(k,pk)

# R=8
# w_kR=3./(ks*R)**3*(np.sin(ks*R)-ks*R*np.cos(ks*R))
# d_sigma2=pk_fun(ks)*w_kR**2*ks**2/(2*np.pi**2)
# sigma8=np.sqrt(integrate.simps(d_sigma2,ks))

##########################################################

#################  Calculation of bv  ###################
class bias_v:

	def __init__(self,z_re,z_obs):
		self.z_obs = z_obs
		self.z_re = z_re
		self.delta_crit = 1.686 # critcal overdensity for collapse
		self.Mhalos=np.logspace(7,16,10000)
		self.ks=np.logspace(-4,3,1000)
		# if (self.z_re-self.z_obs)>6.0:
		# 	raise ParamError('z_re - z_obs should be smaller than 6.0')

		self.halo_func()

	def Pk(self):
		data=np.loadtxt('z'+str(self.z_obs)+'_pk.dat')
		k = data[:,0]
		pk = data[:,1]
		pk_fun=interpolate.interp1d(k,pk)

		return pk_fun

	def halo_func(self):
		pk_fun=self.Pk()

		A=0.186
		a=1.47
		b=2.57
		c=1.19

		sigmas=[]
		for m in self.Mhalos:
			R=(3.*m/4./np.pi/rhom)**(1./3)
			w_kR=3./(self.ks*R)**3*(np.sin(self.ks*R)-self.ks*R*np.cos(self.ks*R))
			d_sigma2=pk_fun(self.ks)*w_kR**2*self.ks**2/(2*np.pi**2)
			sigma_2=integrate.simps(d_sigma2,self.ks)
			sigma=np.sqrt(sigma_2)  
			sigmas.append(sigma)

		sigmas=np.array(sigmas)
		self.sigmas=sigmas
		f_sigma=A*((sigmas/b)**(-a)+1)*np.exp(-c/sigmas**2)

		self.dndM=f_sigma*rhom/self.Mhalos*np.gradient(np.log(sigmas**(-1)),self.Mhalos)


	# HI abundance
	def M_b(self,Mhalo,z_re,file,xi_arr):
		f_b = 0.17
		MF=pd.read_csv(file,index_col='z_obs')
		# MF_nv=pd.read_csv(file_nv,index_col='z_obs')
		z_res=[6.0,7.0,8.0,8.5,9.0,10.0,11.0,12.0]

		z_range=np.arange(6.0,z_re+0.02,0.01)
		xi_dot=np.gradient(xi_arr(z_range),z_range)
		# xi_dot_nv=np.gradient(xi_arr_nov(z_range),z_range)
		mf = interpolate.interp1d(np.array(z_res),MF.loc[self.z_obs].to_numpy(),fill_value="extrapolate")
		# mf_nv = interpolate.interp1d(np.array(z_res),MF_nv.loc[self.z_obs].to_numpy())
		Mb = -integrate.simps(f_b*Mhalo*(1+(2**(1./3)-1)*mf(z_range)/Mhalo)**(-3.)*xi_dot,z_range)
		# Mb_nint = integrate.simps(f_b*Mhalo*(1+(2**(1./3)-1)*mf_nv(z_range)/Mhalo)**(-3.)*xi_dot_nv(z_range))

		Mb_noint = f_b*Mhalo*(1+(2**(1./3)-1)*MF.loc[self.z_obs][str(z_re)]/Mhalo)**(-3.)
		return Mb,Mb_noint

	def rho_HI_direct(self,file_sv,file_nv,xi_arr,xi_arr_nov):
		# f_b = 0.17
		# MF_sv=pd.read_csv(file_sv,index_col='z_obs')
		# MF_nv=pd.read_csv(file_nv,index_col='z_obs')
		# rho_HI_sv=[]
		# rho_HI_nv=[]
		# z_res=[6.0,7.0,8.0,8.5,9.0,10.0,11.0,12.0]

		self.M_baryon = np.array([self.M_b(mh,self.z_re,file_sv,xi_arr)[0] for mh in self.Mhalos])
		self.M_baryon_nv = np.array([self.M_b(mh,self.z_re,file_nv,xi_arr_nov)[0] for mh in self.Mhalos])
		# M_baryon_sv = self.Mhalos*integrate.simps(f_b*(1+(2**(1./3)-1)*mf_sv(z_range)/self.Mhalos)**(-3.)*xi_dot(z_range),z_range)
		rho_HI_sv=integrate.simps(self.dndM*self.M_baryon,self.Mhalos)
		# M_baryon_nv = self.Mhalos*integrate.simps(f_b*(1+(2**(1./3)-1)*mf_nv(z_range)/self.Mhalos)**(-3.)*xi_dot_nv(z_range),z_range)
		rho_HI_nv=integrate.simps(self.dndM*self.M_baryon_nv,self.Mhalos)

		bv=np.diff([rho_HI_nv,rho_HI_sv])/np.mean([rho_HI_nv,rho_HI_sv])
		
		return bv[0]

	def rho_HI(self,file_name,xi_arr):
		f_b = 0.17
		MF=pd.read_csv(file_name,index_col='z_obs')
		rho_HI=[]
		rho_HI_noint = []
		z_res=[6.0,7.0,8.0,8.5,9.0,10.0,11.0,12.0]
		# z_res=[11.0,12.0]

		for z_re in z_res:
			print('z_re=%f' % z_re)
			# z_range=np.arange(6.0,z_re+0.2,0.01)
			# xi_dot=np.gradient(xi_arr(z_range),z_range)

			M_baryon = np.array([self.M_b(mh,z_re,file_name,xi_arr)[0] for mh in self.Mhalos])
			M_baryon_noint = np.array([self.M_b(mh,z_re,file_name,xi_arr)[1] for mh in self.Mhalos])
			rho_HI.append(integrate.simps(self.dndM*M_baryon,self.Mhalos))
			rho_HI_noint.append(integrate.simps(self.dndM*M_baryon_noint,self.Mhalos))

		rho_HI_itp = interpolate.interp1d(np.array(z_res),np.array(rho_HI))
		rho_HI_itp_noint = interpolate.interp1d(np.array(z_res),np.array(rho_HI_noint))


		if self.z_obs<6.0: z_low = 6.0
		else: z_low = self.z_obs
		z_range=np.arange(z_low,self.z_re+0.01,0.01)
		rhos=np.array(rho_HI_itp(z_range))
		rhos_noint=np.array(rho_HI_itp_noint(z_range))
		xi_dot=np.gradient(xi_arr(z_range),z_range)

		# print(xi_arr(z_range))
		# print(xi_dot)

		rho_HI=integrate.simps(-rhos*xi_dot,z_range)
		rho_HI_noint=integrate.simps(-rhos_noint*xi_dot,z_range)
		# print(rho_HI)
		# print(integrate.simps(-rhos,xi_arr(z_range)))

		return rho_HI,rho_HI_noint

	def bias1_func(self):
		# f_b=0.1573
		# MF=pd.read_csv('fmass_mean-sv.csv',index_col='z_obs')
		# z_res=[6.0,7.0,8.0,8.5,9.0,10.0,11.0,12.0]
		# z_range=np.arange(6.0,self.z_re+0.01,0.01)
		# xi_dot=np.gradient(xi_arr(z_range),z_range)

		# mf=interpolate.interp1d(np.array(z_res),MF.loc[self.z_obs].to_numpy)
		# self.M_baryon = integrate.simps(f_b*self.Mhalos*(1+(2**(1./3)-1)*mf(z_range)/self.Mhalos)**(-3.)*xi_dot,z_range)

		# self.M_baryon = np.array([self.M_b(mh,self.z_re,'fmass_mean-sv.csv',xi_arr) for mh in self.Mhalos])

		sigma=self.sigmas
		y=np.log10(200)
		A=1+0.24*y*np.exp(-(4./y)**4) #bias equation parameters
		a=0.44*y-0.84
		B=0.183
		b=1.5
		C=0.019+0.107*y+0.19*np.exp(-(4./y)**4)
		c=2.4
		nu=self.delta_crit/sigma
		bias=1-A*nu**a/(nu**a+self.delta_crit**a)+B*nu**b+C*nu**c

		b1=integrate.simps(self.dndM*bias*self.M_baryon,self.Mhalos)/integrate.simps(self.dndM*self.M_baryon,self.Mhalos)
		bs2=integrate.simps(self.dndM*(-2./7)*(bias-1)*self.M_baryon,Mhalos)/integrate.simps(self.dndM*self.M_baryon,Mhalos)

		return b1,bs2


	def bias2_func(self):
		# a and p are parameters for ST mass function
		sigma=self.sigmas
		a = 0.707
		p = 0.3
		nu=self.delta_crit/sigma

		bias=8./21*((a*nu**2-1)/self.delta_crit+2*p/self.delta_crit/(1+(a*nu**2)**p))/self.delta_crit+(nu**2-3)/sigma**2+2*p/self.delta_crit**2/(1+(a*nu**2)**p)*(2*p+2*a*nu**2-1)
		b2=integrate.simps(self.dndM*bias*self.M_baryon,self.Mhalos)/integrate.simps(self.dndM*self.M_baryon,self.Mhalos)

		return b2


# data=np.loadtxt('z5.5_pk.dat')

# k=data[:,0]
# pk=data[:,1]
# pk_fun=interpolate.interp1d(k,pk)
# for m in Mhalos:
# 	R=(3.*m/4./np.pi/rhom)**(1./3)
# 	w_kR=3./(ks*R)**3*(np.sin(ks*R)-ks*R*np.cos(ks*R))
# 	d_sigma2=pk_fun(ks)*w_kR**2*ks**2/(2*np.pi**2)
# 	sigma_2=integrate.simps(d_sigma2,ks)
# 	sigma=np.sqrt(sigma_2)  
# 	sigmas.append(sigma)

# sigmas=np.array(sigmas)
# f_sigma=A*((sigmas/b)**(-a)+1)*np.exp(-c/sigmas**2)

# dndM=f_sigma*rhom/Mhalos*np.gradient(np.log(sigmas**(-1)),Mhalos)

# M_baryon=[f_b*Mhalos*(1+(2**(1./3)-1)*mf/Mhalos)**(-3.) for mf in MF]
# rho_test=integrate.simps(dndM*Mhalos,Mhalos) 
 

# sigmas_t=np.linspace(0.0001,np.max(sigmas),10000)
# f_sigma_test=A*((sigmas_t/b)**(-a)+1)*np.exp(-c/sigmas_t**2)
# sigma_fraction=integrate.simps(f_sigma_test/sigmas_t,sigmas_t) # -integrate.simps(f_sigma/sigmas,sigmas)

# rho_HI=[integrate.simps(dndM*mb,Mhalos) for mb in M_baryon]

# bv=np.abs(np.diff(rho_HI))/np.mean(rho_HI)
# print('bv='+str(bv[0]))

#################  Calculation of b1  ###################
# delta_crit=1.686 # critcal overdensity for collapse

# def bias1_func(sigma):
# 	y=np.log10(200)
# 	A=1+0.24*y*np.exp(-(4./y)**4) #bias equation parameters
# 	a=0.44*y-0.84
# 	B=0.183
# 	b=1.5
# 	C=0.019+0.107*y+0.19*np.exp(-(4./y)**4)
# 	c=2.4
# 	nu=delta_crit/sigma
# 	bias=1-A*nu**a/(nu**a+delta_crit**a)+B*nu**b+C*nu**c
# 	return bias


# def bias2_func(sigma):
# 	# a and p are parameters for ST mass function
# 	a = 0.707
# 	p = 0.3
# 	nu=delta_crit/sigma
# 	return 8./21*((a*nu**2-1)/delta_crit+2*p/delta_crit/(1+(a*nu**2)**p))/delta_crit+(nu**2-3)/sigma**2+2*p/delta_crit**2/(1+(a*nu**2)**p)*(2*p+2*a*nu**2-1)


# b1=integrate.simps(dndM*bias1_func(sigmas)*np.array(M_baryon[0]),Mhalos)/integrate.simps(dndM*np.array(M_baryon[0]),Mhalos)
# b2=integrate.simps(dndM*bias2_func(sigmas)*np.array(M_baryon[0]),Mhalos)/integrate.simps(dndM*np.array(M_baryon[0]),Mhalos)
# bs2=integrate.simps(dndM*(-2./7)*(bias1_func(sigmas)-1)*np.array(M_baryon[0]),Mhalos)/integrate.simps(dndM*np.array(M_baryon[0]),Mhalos)

# print('b1='+str(b1))
# print('b2='+str(b2))
# print('bs2='+str(bs2))
#######################################################################

class b_sink:

	def __init__(self,  z_re, z_obs, scenario=None):
		self.H_0=67.74e3/(3.0857e22) # Hubble constants now, 67.74 km/s/mpc
		self.Omega_m=0.3089 # Omega_m = 0.27+-0.04
		self.z_obs = z_obs
		self.z_re = z_re
		self.scenario = scenario if scenario is not None else 'fiducial'
		self.z_range=self.z_re-self.z_obs
		self.dz = -0.01
		self.Yp = 0.249 # Helium abudance
		self.rho_crit=1.88e-29 # h^2 g/cm^3
		Omega_mh2 = 0.0223
		mH = 1.6729895e-24 # g
		self.sigma_bc = 33 # RMS streaming velocity 33 km/s
		self.nH_0 = (1-self.Yp)*self.rho_crit*Omega_mh2/mH*(3.0857e24)**3 # cosmological mean value of hydrogen at present, [Mpc^(-3)]
		self.ne_0 = self.rho_crit*Omega_mh2/mH # cosmological mean value of electron at present, [cm^(-3)]

		self.fold_cr="/users/PCON0003/osu10670/bao_21cm/cr-1/"
		self.fold_cr_nov="/users/PCON0003/osu10670/bao_21cm/cr-nov-1/"

		self.CR = self.extract_CRs(self.fold_cr)
		self.alphaB = self.extract_alphaBs(self.fold_cr)
		self.CR_nov = self.extract_CRs(self.fold_cr_nov)
		self.alphaB_nov = self.extract_alphaBs(self.fold_cr_nov)

		self.xi_arr, self.xi_dots, self.CR_mean = self.xi_his(self.CR, self.alphaB)
		self.xi_arr_nov, self.xi_dots_nov, self.CR_mean_nov = self.xi_his(self.CR_nov, self.alphaB_nov)

		bsinks=self.xi_arr_nov(np.arange(self.z_obs,self.z_re,0.1))-self.xi_arr(np.arange(self.z_obs,self.z_re,0.1))
		
	def xi_func(self):
		return self.xi_arr, self.xi_arr_nov
		# return self.xi_arr, self.xi_arr_nov
		# print(bsinks)

		# print(self.xi_arr(np.arange(z_obs,z_re,0.1)))
		# print(self.xi_arr_nov(np.arange(z_obs,z_re,0.1)))

		# print(self.CR_mean(np.arange(z_obs,z_re,0.1)))
		# print(self.CR_mean_nov(np.arange(z_obs,z_re,0.1)))

		# print(self.xi_dots(np.arange(z_obs,z_re,0.1)))
		# print(self.xi_dots_nov(np.arange(z_obs,z_re,0.1)))

		# plt.plot(self.xi_arr(np.arange(z_obs,z_re,0.1)),bsinks**2) 
		# plt.ylabel(r'$b^2_{sinks}$')
		# plt.xlabel(r'$x_i$')
		# plt.yscale('log')
		# plt.show()
		
		# alphaB=np.array([self.alphaB(z_re,z) for z in np.arange(z_re-self.z_obs,0,-0.01)])
		# alphaB_nov=np.array([self.alphaB_nov(z_re,z) for z in np.arange(z_re-self.z_obs,0,-0.01)])

		# plt.plot(np.arange(z_obs,z_re,0.01),alphaB,label='alphaB') 
		# plt.plot(np.arange(z_obs,z_re,0.01),alphaB_nov,label='alphaB_nov')
		# plt.ylabel('alphaB')
		# plt.xlabel('z')
		# plt.legend()
		# plt.show()

		CR=np.array([self.CR(z_re,z) for z in np.arange(z_re-self.z_obs,0,-0.01)])
		CR_nov=np.array([self.CR_nov(z_re,z) for z in np.arange(z_re-self.z_obs,0,-0.01)])

		# plt.plot(np.arange(z_obs,z_re,0.01),CR,label='CR') 
		# plt.plot(np.arange(z_obs,z_re,0.01),CR_nov,label='CR_nov')
		# plt.ylabel('CR')
		# plt.xlabel('z')
		# plt.yscale('log')
		# plt.legend()
		# plt.show()



		# plt.plot(np.arange(z_obs,z_re,0.01),self.xi_arr(np.arange(z_obs,z_re,0.01)),label='xi')
		# plt.plot(np.arange(z_obs,z_re,0.01),self.xi_arr_nov(np.arange(z_obs,z_re,0.01)),label='xi-nov')
		# # plt.plot(np.arange(z_obs,z_re,0.01),self.xi_arr(np.arange(z_obs,z_re,0.01))-self.xi_arr_nov(np.arange(z_obs,z_re,0.01)),label='xi-xi_nov')
		# plt.ylabel('xi')
		# plt.xlabel('z')
		# plt.yscale('log')
		# plt.legend()
		# plt.show()

		# plt.plot(np.arange(z_obs,z_re,0.01),self.xi_dots(np.arange(z_obs,z_re,0.01)),label=r'$\dot{x_i}$')
		# plt.plot(np.arange(z_obs,z_re,0.01),self.xi_dots_nov(np.arange(z_obs,z_re,0.01)),label=r'$\dot{x_i}_{nov}$')
		# plt.ylabel(r'$\dot{x_i}$')
		# plt.xlabel('z')
		# plt.legend()
		# plt.show()

	def Clump(self):
		cr=[self.CR_mean(z) for z in np.arange(self.z_obs,self.z_re,0.01)]
		cr_nov=[self.CR_mean_nov(z) for z in np.arange(self.z_obs,self.z_re,0.01)]
		return cr, cr_nov

	def extract_CRs(self,dir):
		CRs =[]
		# for fold in os.listdir(dir):
		if dir==self.fold_cr : folds = ['HMZ08-cr','HMZ8.5-cr','HMZ09-cr','HMZ10-cr','HMZ11-cr','HMZ12-cr']
		if dir==self.fold_cr_nov : folds = ['HMZ08-nov-cr','HMZ8.5-nov-cr','HMZ09-nov-cr','HMZ10-nov-cr','HMZ11-nov-cr','HMZ12-nov-cr']
		for fold in folds:
			# get index for z=7.9
			# with open(dir+fold+"/outputs.txt", 'r') as f: 
			# 	ind = f.read()
			# 	ind = [float(i) for i in ind.split()]
			# 	la_ind = len(ind)
			# 	ind = ind.index(0.112359550562)

			CR = []
			CR.append(float(re.findall('[0-9.]+',fold)[0])) #[z_re, CRs at this z_re]
			# for i in range(ind,la_ind):
			for i in np.arange(0,60,1):
				with open(dir+fold+"/snapshot_0"+str(i).zfill(2)+".clump2", 'r') as f: 
					data=np.array(f.read().split(),dtype=float)
				CR.append(data[0])
			CRs.append(CR)

		CRs.sort(key=lambda x: x[0])
		CRs=np.array(CRs)

		pos=zip(*np.where(np.isnan(CRs)==True)) # interpolate NAN value
		for i in pos:
			if 1<i[1]<len(CRs[0])-1:
				CRs[i]=(CRs[i[0],i[1]-1]+CRs[i[0],i[1]+1])/2

		coordinates =[]
		x = CRs[:,0]
		y = np.arange(0,6.,0.1) # z_re-z as y axis

		for i in x:
			for j in y:
				coordinates.append((i,j))

		Z_CR = np.delete(CRs,0,1).flatten()
		# print(coordinates)
		# print(Z_CR)
		# X, Y = np.meshgrid(x,y)
		f_CR = interpolate.LinearNDInterpolator(coordinates, Z_CR, fill_value=0)

		return f_CR

	def extract_alphaBs(self,dir):
			alphaBs =[]
			# for fold in os.listdir(dir):
			if dir==self.fold_cr : folds = ['HMZ08-cr','HMZ8.5-cr','HMZ09-cr','HMZ10-cr','HMZ11-cr','HMZ12-cr']
			if dir==self.fold_cr_nov : folds = ['HMZ08-nov-cr','HMZ8.5-nov-cr','HMZ09-nov-cr','HMZ10-nov-cr','HMZ11-nov-cr','HMZ12-nov-cr']
			for fold in folds:
				alphaB = []
				alphaB.append(float(re.findall('[0-9.]+',fold)[0])) #[z_re, CRs at this z_re]
				# for i in range(ind,la_ind):
				for i in np.arange(0,60,1):
					with open(dir+fold+"/snapshot_0"+str(i).zfill(2)+".clump2", 'r') as f: 
						data=np.array(f.read().split(),dtype=float)
					alphaB.append(data[1])
				alphaBs.append(alphaB)

			alphaBs.sort(key=lambda x: x[0])
			alphaBs=np.array(alphaBs)

			pos=zip(*np.where(np.isnan(alphaBs)==True)) # interpolate NAN value
			for i in pos:
				if 1<i[1]<len(CRs[0])-1:
					alphaBs[i]=(CRs[i[0],i[1]-1]+alphaBs[i[0],i[1]+1])/2

			coordinates =[]
			x = alphaBs[:,0]
			y = np.arange(0,6.,0.1) # z_re-z as y axis

			for i in x:
				for j in y:
					coordinates.append((i,j))

			Z_alphaB = np.delete(alphaBs,0,1).flatten()
			# print(coordinates)
			# print(Z_alphaB)
			# X, Y = np.meshgrid(x,y)
			f_alphaB = interpolate.LinearNDInterpolator(coordinates, Z_alphaB, fill_value=0)

			return f_alphaB

	def emissivity(self,z): # s^-1
		f_esc = 0.2
		epsilon_ion = pow(10,53.14)

		ap=0.01306
		bp=3.66
		cp=2.28
		dp=5.29

		Mpc_cm_factor = 3.0857e24
		rho_SFR=ap*((1+z)**bp)/(1+((1+z)/cp)**dp)

		if self.scenario == 'fiducial': 
			epsilon=f_esc*epsilon_ion*rho_SFR
		elif self.scenario == 'delay':
			epsilon=f_esc*epsilon_ion*rho_SFR/1.55
		return epsilon

	def nH(self,z): # mean value of hydrogen number density
		return self.nH_0*(1+z)**3

	def ne(self,z): # mean value of electron number density
		return self.ne_0*(1+z)**3

	def xi_his(self, CRs, alphaBs):
		xi_arr=[0]
		xi = 0
		xi_dots = [self.emissivity(self.z_re)/self.nH_0]
		xi_dot = self.emissivity(self.z_re)/self.nH_0
		CR_mean=[0]
		for z in np.arange(self.z_re,self.z_obs+self.dz,self.dz):
			xi += -xi_dot*self.dz/(self.H_0*np.sqrt(self.Omega_m))*((1+z)**(-5./2))
			if xi >=1. : xi = 1. # reionization completed
			xi_arr.append(xi)
			CR=0
			for z_r in np.arange(self.z_re,z+self.dz,self.dz):
				CR+=CRs(self.z_re,z_r-z)*(xi_arr[int((z_r-self.z_re)/self.dz)+1]-xi_arr[int((z_r-self.z_re)/self.dz)])
			CR /= xi
			CR_mean.append(CR)
			xi_dot = self.emissivity(z)/self.nH_0-alphaBs(self.z_re,self.z_re-z)*self.ne(z)*CR*xi #n_H is comoving density, ne is proper density
			xi_dots.append(xi_dot)

		# print(np.array(-self.alphaB(self.z_re,z)*self.ne_0/((3.0857e24)**3)*np.array(CR_mean)*np.array(xi_arr)))
		# print(np.array(xi_arr))
		# print(np.array(xi_dots))
		# print(np.array(CR_mean))
		xi_arr = interpolate.interp1d(np.arange(self.z_re, self.z_obs+2*self.dz,self.dz), np.array(xi_arr),fill_value="extrapolate")
		xi_dots = interpolate.interp1d(np.arange(self.z_re, self.z_obs+2*self.dz,self.dz), np.array(xi_dots),fill_value="extrapolate")
		CR_mean = interpolate.interp1d(np.arange(self.z_re, self.z_obs+2*self.dz,self.dz), np.array(CR_mean),fill_value="extrapolate")

		return xi_arr, xi_dots, CR_mean

	def dxi_dz(self):
		b=self.xi_arr_nov(np.arange(self.z_obs,self.z_re,0.1))-self.xi_arr(np.arange(self.z_obs,self.z_re,0.1))

		return b

	def A(self, zt):
#		z = (3*self.H_0*self.Omega_m**(1./2.)*t/2.)**(-2./3.)-1
		# r = np.arange(self.z0,self.z_obs,self.dz)
		# return np.mean(self.xi_dots(r))/np.mean(self.xi_arr(r))+self.alphaB(self.z_re,zt)*self.ne(zt)*self.CR_mean(zt)
		return self.xi_dots_nov(zt)/self.xi_arr_nov(zt)+self.alphaB_nov(self.z_re,self.z_re-zt)*self.ne(zt)*self.CR_mean_nov(zt)


	def B(self, zt):
#		z = (3*self.H_0*self.Omega_m**(1./2.)*t/2.)**(-2./3.)-1
		return -self.alphaB_nov(self.z_re,self.z_re-zt)*self.ne(zt)*self.CR_mean_nov(zt)

	def D(self, zt):
		s = 0.
		for z in np.arange(self.z_re+self.dz, zt+self.dz, self.dz):
			s += -1./self.H_0/np.sqrt(self.Omega_m)*(1+z)**(-5./2)*self.dz*self.A(z)

		return np.exp(-s)

	def F(self, zt):
		s = 0.
		for z in np.arange(self.z_re+self.dz, zt+self.dz, self.dz):
			s += -1./self.H_0/np.sqrt(self.Omega_m)*(1+z)**(-5./2)*self.dz*self.A(z)

		return self.B(zt)*np.exp(s)

	def S(self, zt):
		return self.CR_mean(zt)-self.CR_mean_nov(zt)
		# s=0.
		# for z in np.arange(self.z0, zt+self.dz, self.dz):
		# 	s += (self.xi_arr(z+self.dz)-self.xi_arr(z))*(self.CR(z,z-zt)-self.CR_nov(z,z-zt))
		# s = s / self.xi_arr(self.z_obs)
		# return s


	def b(self):
		s = 0.
		for z in np.arange(self.z_re, self.z_obs, self.dz):
			s += -1./self.H_0/np.sqrt(self.Omega_m)*(1+z)**(-5./2.)*self.dz*self.F(z)*self.S(z)
		
		b_xv = self.D(self.z_obs)/self.CR_mean_nov(self.z_obs)*s

		# r = np.arange(self.z0,self.z_obs,self.dz)
		return -self.xi_arr_nov(self.z_obs)*b_xv


# z = 4.5
# xi= b_sink(9.0,z'fiducial')
# xi_sv, xi_nv = xi.xi_func()
# rho=bias_v(9.0,z)

# Mbs = []
# Mbs_noint = []
# for mh in np.logspace(7,15,10000):
# 	Mb,Mb_noint = rho.M_b(mh,9.0,'fmass_mean.csv',xi_sv)
# 	Mbs.append(Mb)
# 	Mbs_noint.append(Mb_noint)

# tb={'z_obs':z_obs,'rho_sv':rho_sv_list,'rho_sv_noint':rho_sv_noint_list,'rho_nv':rho_nv_list,'rho_nv_noint':rho_nv_noint_list}
# pd.DataFrame(tb).to_csv('rho_HI_table_zobs_'+str(z_obs)+'.csv')


# Check rho_HI integration

# for i in range(4):
# z_obs=[3.5,4.0,4.5,5.0,5.5]
# z_obs=[4.5,5.0,5.5]

# rho_sv_list = []
# rho_sv_noint_list = []
# rho_nv_list = []
# rho_nv_noint_list = []

# for z in z_obs:
# 	xi= b_sink(9.0,z,'fiducial')
# 	xi_sv, xi_nv = xi.xi_func()
# 	rho=bias_v(9.0,z)

# 	rho_sv,rho_sv_noint = rho.rho_HI('fmass_mean.csv',xi_sv)
# 	rho_nv,rho_nv_noint = rho.rho_HI('fmass_mean-nv.csv',xi_nv)
# 	print(rho_sv,rho_sv_noint)
# 	print(rho_nv,rho_nv_noint)
# 	rho_sv_list.append(rho_sv)
# 	rho_sv_noint_list.append(rho_sv_noint)
# 	rho_nv_list.append(rho_nv)
# 	rho_nv_noint_list.append(rho_nv_noint)

# 	tb={'z_obs':z_obs,'rho_sv':rho_sv_list,'rho_sv_noint':rho_sv_noint_list,'rho_nv':rho_nv_list,'rho_nv_noint':rho_nv_noint_list}
# 	pd.DataFrame(tb).to_csv('rho_HI_table_zobs_'+str(z_obs)+'.csv')

# z_obs=[3.5,4.0,4.5,5.0,5.5]
# b1=[]
# b2=[]
# bs2=[]
# bv=[]
# bv_direct=[]
# bv_indirect=[]
# bv_indirect_sink=[]
# dxi_dz=[]

# for z in z_obs:
# 	xi= b_sink(9.0,z,'delay')
# 	xi_sv, xi_nv = xi.xi_func()
# 	rho=bias_v(9.0,z)

# 	rho_sv = rho.rho_HI('fmass_mean.csv',xi_sv)
# 	rho_nv = rho.rho_HI('fmass_mean-nv.csv',xi_nv)

# 	bv_tmp=np.log(rho_sv/rho_nv)
# 	b1_tmp,bs2_tmp=rho.bias1_func()
# 	b2_tmp=rho.bias2_func()
# 	bv_direct_tmp=rho.rho_HI_direct('fmass_mean.csv','fmass_mean-nv.csv')


# 	b1.append(b1_tmp)
# 	b2.append(b2_tmp)
# 	bv.append(bv_tmp)
# 	bs2.append(bs2_tmp)
# 	bv_direct.append(bv_direct_tmp)
# 	bv_indirect.append(bv_tmp-bv_direct_tmp)
# 	bv_indirect_sink.append(xi.b())
# 	dxi_dz.append(np.sum(xi.dxi_dz()))
# 	# print(xi.dxi_dz())
	
# 	print('For delayed reionization')
# 	print('At z_obs='+str(z)+', bv='+str(bv_tmp)+',bv_direct='+ str(bv_direct_tmp)+',bv_ind='+str(bv_tmp-bv_direct_tmp))
# 	print('b1='+str(b1_tmp))
# 	print('bs2='+str(bs2_tmp))
# 	print('b2='+str(b2_tmp))


# Main results
# for i in range(4):
# 	z_obs=[3.5,4.0,4.5,5.0,5.5]
# 	b1=[]
# 	b2=[]
# 	bs2=[]
# 	bv=[]
# 	bv_direct=[]
# 	bv_indirect=[]
# 	bv_indirect_sink=[]
# 	dxi_dz=[]

# 	for z in z_obs:
# 		xi= b_sink(9.0,z,'fiducial')
# 		xi_sv, xi_nv = xi.xi_func()
# 		rho=bias_v(9.0,z)

# 		rho_sv = rho.rho_HI('fmass_'+str(i+1)+'.csv',xi_sv)
# 		rho_nv = rho.rho_HI('fmass_'+str(i+1)+'-nv.csv',xi_nv)

# 		bv_tmp=np.log(rho_sv/rho_nv)
# 		bv_direct_tmp=rho.rho_HI_direct('fmass_'+str(i+1)+'.csv','fmass_'+str(i+1)+'-nv.csv',xi_sv,xi_nv)
# 		b1_tmp,bs2_tmp=rho.bias1_func()
# 		b2_tmp=rho.bias2_func()

# 		b1.append(b1_tmp)
# 		b2.append(b2_tmp)
# 		bv.append(bv_tmp)
# 		bs2.append(bs2_tmp)
# 		bv_direct.append(bv_direct_tmp)
# 		bv_indirect.append(bv_tmp-bv_direct_tmp)
# 		bv_indirect_sink.append(xi.b())
# 		dxi_dz.append(np.sum(xi.dxi_dz()))
# 		# print(xi.dxi_dz())
		
# 		print('For fiducial reionization')
# 		print('At z_obs='+str(z)+', bv='+str(bv_tmp)+',bv_direct='+ str(bv_direct_tmp)+',bv_ind='+str(bv_tmp-bv_direct_tmp))
# 		print('b1='+str(b1_tmp))
# 		print('bs2='+str(bs2_tmp))
# 		print('b2='+str(b2_tmp))

# 	tb={'z_obs':z_obs,'b1':b1,'b2':b2,'bs2':bs2, 'bv':bv, 'bv_dir':bv_direct, 'bv_ind':bv_indirect,'bv_ind_sink':bv_indirect_sink,'dxi_dz':
# 	dxi_dz}
# 	pd.DataFrame(tb).to_csv('bias_table_'+str(i+1)+'_v2.csv')


###### xi history quantity #######
# z_re = 11.9
# bias = b_sink(z_re,z_re-6.0,'delay')
# xi=interpolate.interp1d(bias.xi_arr(np.arange(z_re-6.0,z_re,0.05)),np.arange(z_re-6.0,z_re,0.05))
# xi_nov=interpolate.interp1d(bias.xi_arr_nov(np.arange(z_re-6.0,z_re,0.05)),np.arange(z_re-6.0,z_re,0.05))
# # dxi_dv=xi-xi_nov
# print(bias.xi_arr(np.arange(z_re-6.0,z_re,0.05)))
# print('z='+str(xi(0.5))+" when xi=0.5")
# print('z='+str(xi(1.0))+" when xi=1")
# print(bias.xi_arr(np.arange(6.8,7.0,0.01)))
# # print('z='+str()+" when xi=1")

# dxi_dv=bias.xi_arr(np.arange(6.8,7.0,0.01))-bias.xi_arr_nov(np.arange(6.8,7.0,0.01))
# print(dxi_dv)


# c=3.0e8
# H_0=67.74 # Hubble constants now, 67.74 km/s/mpc
# # H_0=67.74/(parsec*1e3) # Hubble constants, /s/m
# G=4.30091e-9  #6.674×10−11 m3*kg−1*s−2 ### 4.30091(25)×10−3 Mpc*M_solar-1*(km/s)^2
# # G=6.674e-11  #6.674×10−11 m3*kg−1*s−2 ### 4.30091(25)×10−3 Mpc*M_solar-1*(km/s)^2
# solar_m= 1.98847e30 #(1.98847±0.00007)×10^30 kg
# parsec=3.085677581e16 # m per parsec

# rho_crit=3*H_0**2/8/np.pi/G # M_solar/(Mpc)^3
# # rho_crit=3*H_0**2/8/np.pi/G/(1e6)/(parsec*1e6)**2 #kg/m^3

# Omega_m=0.3089 # Omega_m = 0.3089+-0.0062
# rhom=rho_crit*Omega_m #*(1+z)**3
# Yp = 0.2454 # Helium abudance
# f_b=0.17 # baryon fraction
# sigma_T=6.65e-29 # Thomson scattering cross section in m^2
# mH = 1.6729895e-27 # kg

# bias = b_sink(12,0.,'delay')
# dz=0.01
# tau_t=0
# tau_t_int=[]
# for z in np.arange(12.,0.,-0.01):
# 	tau_t_int.append(tau_t)
# 	# if z>6.8:	xi=bias.xi_arr(z)
# 	# else:	xi=1
# 	xi=bias.xi_arr(z)
# 	print([xi,z])
# 	tau_t=Omega_m*f_b*rho_crit*(1-Yp)/mH*(1+Yp/4./(1.-Yp))*xi*(1+z)**3*sigma_T*c/(1+z)/(H_0*np.sqrt(Omega_m*(1+z)**3))*dz*solar_m/(parsec*1e6)**2/1000

# print('Thomson scattering cross section with streaming velocity is '+str(np.sum(tau_t_int)))

# $\tau=0.123,0.096, 0.061,0.026$ with $z_{re}=12,9,8,7$ respectively]$
# plt.plot(np.arange(12.0,0,-0.01),tau_t_int)
# plt.xlabel('z')
# plt.ylabel(r'$d\tau$')
# plt.show()

# tau_t=0
# tau_t_int=[]
# for z in np.arange(12.0,0,-0.01):
# 	tau_t_int.append(tau_t)
# 	if z>6.8:	xi=bias.xi_arr_nov(z)
# 	else:	xi=1
# 	tau_t=Omega_m*rho_crit*(1-Yp)/mH*(1+Yp/4./(1.-Yp))*xi*(1+z)**3*sigma_T*c/(1+z)/(H_0*np.sqrt(Omega_m*(1+z)**3))*dz*solar_m/(parsec*1e6)**2/1000/(H_0/100)

# print('Thomson scattering cross section without streaming velocity is '+str(np.sum(tau_t_int)))



# z_re=float(sys.argv[1])
# z_obs=float(sys.argv[2])


# CR,CR_nov=bias.Clump()

# plt.plot(np.arange(z_obs,z_re,0.01),CR,label="CR")
# plt.plot(np.arange(z_obs,z_re,0.01),CR_nov,label='CR_nov')
# plt.yscale('log')
# plt.legend()
# plt.show()

# label=[['',300],['2',100],['3',500]]
# for i in range(3):
# 	bias = b_sink(z_re,z_obs,label[i][0])
# 	b= bias.dxi_dz()
# 	xi=bias.xi_arr(np.arange(z_obs,z_re,0.1))
# 	plt.plot(xi,b**2,label=r'$\rho_{cutoff}=$'+str(label[i][1])+r'$\rho_{mean}$')

# plt.legend()
# plt.yscale('log')
# plt.xlabel(r'$x_i$')
# plt.ylabel(r'$b^2_{sink}$')
# plt.show()
# plt.close()

# label=[['',300],['2',100],['3',500]]


# fig, (ax1, ax2) = plt.subplots(2, 1)
# z_re = 12.0
# bias = b_sink(z_re,z_re-6.5,'fiducial')
# xi=bias.xi_arr(np.arange(z_re-6.5,z_re,0.1))
# xi_nov=bias.xi_arr_nov(np.arange(z_re-6.5,z_re,0.1))
# dxi_dz=-np.gradient(xi,np.arange(z_re-6.5,z_re,0.1))
# dxi_dv=xi-xi_nov

# bias_del = b_sink(z_re,z_re-6.5,'delay')
# xi_del=bias_del.xi_arr(np.arange(z_re-6.5,z_re,0.1))

# z_c=np.where(xi<1.0)[0][0]
# z_c_del=np.where(xi_del<1.0)[0][0]
# ax1.plot(np.arange(z_re-6.5+z_c_del*0.1,z_re,0.1),xi[z_c_del:],color='red',linewidth=0.8,label='Fiducial')
# ax1.plot(np.arange(z_re-6.5+z_c_del*0.1,z_re,0.1),xi_del[z_c_del:],color='blue',linewidth=0.8,label='Delayed Reionization')
# # ax2.plot(np.arange(z_re-6.0,z_re,0.1),dxi_dz)
# ax2.plot(np.arange(z_re-6.5+z_c*0.1,z_re,0.1),dxi_dv[z_c:],linewidth=0.8, color='red')
# ax2.set_xlim([z_re-6.5+z_c_del*0.1,z_re])

# xi=bias.xi_arr(np.arange(z_re-6.5,z_re,0.1))
# xi_nov=bias.xi_arr_nov(np.arange(z_re-6.5,z_re,0.1))
# dxi_dz=-np.gradient(xi,np.arange(z_re-6.5,z_re,0.1))
# dxi_dv=xi-xi_nov

# ax1.set_yscale('log')
# ax1.set_ylabel(r'$x_i$')
# ax1.set(title='Reionization Histories')
# ax1.legend()
# # ax2.set_yscale('log')
# # ax2.set_ylabel(r'$dx_i/dz$')
# # ax2.set_yscale('log')
# ax2.set_yscale('log')
# ax2.set_xlabel('z')
# ax2.set_ylabel(r'$\partial x_i/\partial(v_{bc}^2/\sigma^2)$')
# plt.figure(figsize=(40,20))
# plt.show()


########## Reionization history plot #############

# fig, (ax1, ax2) = plt.subplots(2, 1)
# z_re = 9.0
# bias = b_sink(z_re,z_re-5.5,'fiducial')
# xi=bias.xi_arr(np.arange(z_re-5.5,z_re,0.1))
# xi_nov=bias.xi_arr_nov(np.arange(z_re-5.5,z_re,0.1))
# dxi_dz=-np.gradient(xi,np.arange(z_re-5.5,z_re,0.1))
# dxi_dv=xi-xi_nov

# bias_del = b_sink(z_re,z_re-5.5,'delay')
# xi_del=bias_del.xi_arr(np.arange(z_re-5.5,z_re,0.1))

# z_c=np.where(xi<1.0)[0][0]
# z_c_del=np.where(xi_del<1.0)[0][0]
# ax1.plot(np.arange(z_re-5.5+z_c_del*0.1,z_re,0.1),xi[z_c_del:],color='red',linewidth=0.8,label='Fiducial')
# ax1.plot(np.arange(z_re-5.5+z_c_del*0.1,z_re,0.1),xi_del[z_c_del:],color='blue',linewidth=0.8,label='Delayed Reionization')
# # ax2.plot(np.arange(z_re-6.0,z_re,0.1),dxi_dz)
# ax2.plot(np.arange(z_re-5.5+z_c*0.1,z_re,0.1),dxi_dv[z_c:],linewidth=0.8, color='red')
# ax2.set_xlim([z_re-5.5+z_c_del*0.1,z_re])

# xi=bias.xi_arr(np.arange(z_re-5.5,z_re,0.1))
# xi_nov=bias.xi_arr_nov(np.arange(z_re-5.5,z_re,0.1))
# dxi_dz=-np.gradient(xi,np.arange(z_re-5.5,z_re,0.1))
# dxi_dv=xi-xi_nov

# ax1.set_yscale('log')
# ax1.set_ylabel(r'$x_i$')
# ax1.set(title='Reionization Histories')
# ax1.legend()
# # ax2.set_yscale('log')
# # ax2.set_ylabel(r'$dx_i/dz$')
# # ax2.set_yscale('log')
# ax2.set_yscale('log')
# ax2.set_xlabel('z')
# ax2.set_ylabel(r'$\partial x_i/\partial(v_{bc}^2/\sigma^2)$')
# plt.figure(figsize=(40,20))
# plt.show()


# for z_re in [8.0,8.5,9.0,10.0,11.0,12.0]:
# 	bias = b_sink(z_re,z_re-6.0)
# 	xi=bias.xi_arr(np.arange(6.0,z_re,0.1))
# 	plt.plot(np.arange(6.0,z_re,0.1),xi,color='red',linewidth=0.8,label=r'$z_{re}=$'+str(z_re))

# plt.yscale('log')
# plt.legend()
# plt.xlabel('z')
# plt.ylabel(r'$x_i$')
# plt.title('Reionization Histories')
# plt.show()

# z_re=9.0
# # for z_re in [6.0,7.0,8.0,8.5,9.0,10.0,11.0,12.0]:
# bias = b_sink(z_re,z_re-6.0)
# xi=bias.xi_arr(np.arange(z_re-6.0,z_re,0.1))
# xi_nov=bias.xi_arr_nov(np.arange(z_re-6.0,z_re,0.1))
# dxi_dz=xi-xi_nov
# plt.plot(np.arange(z_re-6.0,z_re,0.1),dxi_dz,label=r'$z_{re}=$'+str(z_re))

# plt.yscale('log')
# plt.legend()
# plt.xlabel('z')
# plt.ylabel(r'$dx_i/dz$')
# plt.title('Reionization Histories')
# plt.show()

##############################################
#########       xi, CR subplots      #########
##############################################
# fig, (ax1, ax2) = plt.subplots(2, 1)
# bias = b_sink(12,6.0)
# xi=bias.xi_arr(np.arange(6.0,12.0,0.1))
# xi_nov=bias.xi_arr_nov(np.arange(6.0,12.0,0.1))
# ax1.plot(np.arange(6.0,12.0,0.1),xi,label='with streaming velocity')
# ax1.plot(np.arange(6.0,12.0,0.1),xi_nov,label='without streaming velocity')


# ax1.set_yscale('log')
# ax1.set_ylabel(r'$x_i$')
# ax1.legend()

# for z_re in [6.0,7.0,8.0,8.5,9.0,10.0,11.0,12.0]:
# 	bias = b_sink(z_re, 6.0)
# 	CR = bias.CR(z_re,np.arange(0.,6.,0.1))
# 	ax2.plot(np.arange(z_re,z_re-6.,-0.1),CR,label=r'$z_{re}=$'+str(z_re))

# ax2.set_yscale('log')
# ax2.set_xlabel('z')
# ax2.set_ylabel(r'$C_R$')
# ax2.legend()

# plt.show()

# b_sink = bias.b()

# print (b_sink) # -0.05971185193008316

##############################################
#########       CR 2D color map      #########
##############################################
# bias = b_sink(z_re,z_obs,'2')
# x = np.arange(8.,12.,0.1)
# y = np.arange(0.,6.0,0.1)
# X, Y = np.meshgrid(x,y)

# Z = bias.CR(X, Y)

# im = plt.imshow(Z, cmap='Blues', extent=(8.,12.,0.,6.), interpolation='bilinear')
# plt.colorbar(im)
# plt.show()

##############################################
#######        CR z_re-z curve          ######
##############################################
# z_res = [8.0,8.5,9.0,10.0,11.0,12.0]
# colors = cm.Set2(np.linspace(0,1,len(z_res)))
# for i,color in enumerate(colors):
# 	bias = b_sink(z_res[i], 6.0)
# 	CR = bias.CR(z_res[i],np.arange(0.,6.,0.1))
# 	plt.plot(np.arange(z_res[i],z_res[i]-6.,-0.1),CR,linewidth=0.8,color=color,label=r'$z_{re}=$'+str(z_res[i]))

# plt.yscale('log')
# plt.legend()
# plt.xlabel('z')
# plt.ylabel(r'$C_R$')
# plt.figure(figsize=(40,20))
# plt.show()



# CRs=[]
# alphaBs=[]
# # get index for z=7.9
# for fold in os.listdir("/fs/scratch/PCON0003/osu10670/cr-nov"):
# 	CR=[]
# 	alphaB = []
# 	with open("/fs/scratch/PCON0003/osu10670/cr-nov/"+fold+"/outputs.txt", 'r') as f:
# 		ind = f.read()
# 	ind = [float(i) for i in ind.split()]
# 	la_ind = len(ind)
# 	ind = ind.index(0.112359550562)
# 	CR.append(float(re.findall('[0-9.]+',fold)[0]))#[z_re, CRs at this z_re]
# 	alphaB.append(float(re.findall('[0-9.]+',fold)[0]))#[z_re, CRs at this z_re]
# 	for i in range(ind,ind+25):
# 		with open("/fs/scratch/PCON0003/osu10670/cr-nov/"+fold+"/snapshot_0"+str(i).zfill(2)+".clump", 'r') as f: 
# 			data=np.array(f.read().split(),dtype=float)
# 			# cr=float(f.read().split()[0])
# 			# alpha=float(f.read().split()[1])
# 		CR.append(data[0])
# 		alphaB.append(data[1])
# 	CRs.append(CR)
# 	alphaBs.append(alphaB)

# CRs.sort(key=lambda x: x[0])
# alphaBs.sort(key=lambda x: x[0])
# CRs=np.array(CRs)
# alphaBs=np.array(alphaBs)

# pos=zip(*np.where(np.isnan(CRs)==True)) # interpolate NAN value
# for i in pos:
# 	if 1<i[1]<len(CRs[0])-1:
# 		CRs[i]=(CRs[i[0],i[1]-1]+CRs[i[0],i[1]+1])/2
# 		alphaBs[i]=(alphaBs[i[0],i[1]-1]+alphaBs[i[0],i[1]+1])/2

# coordinates =[]
# x = CRs[:,0]
# y = np.arange(7.9,5.4,-0.1)

# for i in x:
# 	for j in y:
# 		coordinates.append((i,j))

# Z_CR = np.delete(CRs,0,1).flatten()
# Z_alpha = np.delete(alphaBs,0,1).flatten()
# f_CR = interpolate.LinearNDInterpolator(coordinates, Z_CR, fill_value='extrapolate')


# import os
# dir = '/fs/scratch/PCON0003/osu10670/cr-nov'
# for fold in os.listdir(dir):
# 	os.system("perl analyze_clump.pl "+dir+"/"+fold)


#sigma8 
#b1=1.88121186

#bv=1.02984052e-05
# sigma_fraction=0.6954724286292236, rho_test/rhom=0.6954732965847203
#diff=8.67894097145161e-07

# test: cluster_toolkit is not validable
# dndM_tool=massfunction.dndM_at_M(Masses, k, pk, Omega_m)
# M_baryon_test=[Masses*(1+(2**(1./3)-1)*mf/Masses)**(-3.) for mf in MF]
# rho_test=integrate.simps(dndM_tooltool*Masses,Masses) #np.sum(dndM*np.diff(Mhalos)*(Mhalos[1:]))
# sigma_fraction=integrate.simps(f_sigma/sigmas,sigmas)

# rho_HI_test=[integrate.simps(dndM_tool*mb,Masses) for mb in M_baryon_test]

# bv_test=np.abs(np.diff(rho_HI_test))/np.mean(rho_HI_test)

#[51837342516.37302,51837203621.49072]
#[2.1335311401533696e+24,2.1335311400894562e+24]


#test 
# import matplotlib.pyplot as plt
# plt.plot(ks,w_kR)
# plt.xscale('log')
# plt.show()

#plt.plot(Mhalos,dndM*Mhalos**2/rhom)
# plt.xscale('log')
# plt.yscale('log')

# from cluster_toolkit import massfunction
# import numpy as np
# #Assume that k and P come from somewhere, e.g. CAMB or CLASS
# #Units of k and P are h/Mpc and (Mpc/h)^3
# Mass = 1e14 #Msun/h
# Omega_m = 0.3 #example value
# dndM = massfunction.dndM_at_M(Mass, k, P, Omega_m)
# #Or could also use an array
# Masses = np.logspace(12, 16)
# dndM = massfunction.dndM_at_M(Masses, k, P, Omega_m)






