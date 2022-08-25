import pybullet as p
import pybullet_data as pd
import numpy as np

from numpy import *
import time


class two_dof_rr:
    ## constructor and attributes
    def __init__(self):
     
      self.rob_id = None
      self.num_jts = None
      self.mvbl_jts = None
      self.m1,self.m2,self.l1,self.l2,self.gravity=1,1,1,1,10
      
    
     
      time_delay=np.linspace(0,10,1000,endpoint=False)
      self.dt= 0.001
      self.ex_F = np.zeros([1000,2])
      self.ex_F[100:250] = np.array([200,200])

        
  # initial conditions
      self.jointAccel = np.array([0,0])
      self.jointTorque = np.array([0,0])
      self.jointVelocity = np.array([0,0])
      

      ## setting up pybullet engine
    def createWorld(self):
        
       
        pc = p.connect(p.GUI)
        
        p.setAdditionalSearchPath(pd.getDataPath())
      
        plane_id = p.loadURDF("plane.urdf")
        p.setTimeStep(self.dt)
        p.setPhysicsEngineParameter()
        p.setRealTimeSimulation(1)
        
        
        ### 2link cuiboidal arm is being loaded into the environment
        
        self.rob_id = p.loadURDF('urdf/2link.urdf', useFixedBase = 1)
        self.num_jts = p.getNumJoints(self.rob_id)  # Joints(including floor and base link which is a fixed one..)
        print('Number of Joints in my arm: ',self.num_jts)
        
        ##here the joints is taken n-1 since base has a fixed joint,hence movable will be n-1
        self.mvbl_jts = list(range(1, self.num_jts - 1 ))
        print('movable joints:', self.mvbl_jts,' End-effector: 2')
        
        
 
# development of analytical jacobian for 2 link manipulator
    def Do_Jaco(self,jt):
    
        jointAngle = jt
       
        J11 = -self.l1 * sin(jointAngle[0]) - self.l2 * sin(jointAngle[0] + jointAngle[1])
        J12 = -self.l2 * sin(jointAngle[0] + self.jointAngle[1])
        J21 = self.l1 * cos(jointAngle[0]) + self.l2 * cos(jointAngle[0] + jointAngle[1])
        J22 = self.l2 * cos(jointAngle[0] + jointAngle[1])

        jaco_mat = np.array([
            [J11,J12],
            [J21,J22]
        ])
       
        return jaco_mat
        
        
    def matrices_4_torq(self,pos,vel):
    
        self.jointAngle = pos
        self.jointVelocity = vel

        L1sq , L2sq  = self.l1 ** 2 , self.l2 ** 2
        
        M11 = self.m1 * L1sq + self.m2 * (L1sq + 2 * self.l1 * self.l1 *
                   cos(self.jointAngle[1]) + L2sq )
        M12 = self.m2 * (self.l1 * self.l2 * cos(self.jointAngle[1]) + L2sq)
        M21 = self.m2 * (self.l1 * self.l2 * cos(self.jointAngle[1]) + L2sq )
        M22 = self.m2 * L2sq
        Inertia_Matrix = np.array([
        
            [M11,M12],
            [M21,M22]])
            
        C11 = -self.m2 * self.l1 * self.l2 * sin(self.jointAngle[1]) * (2 *   self.jointVelocity[0] * self.jointVelocity[1] + pow(self.jointVelocity[1],2))
        C12 = self.m2 * self.l1 * self.l2 * pow (self.jointVelocity[0],2) * sin(self.jointAngle[1])
        coriolisMatrix = np.array([C11,C12])
        
        G11 = (self.m1 + self.m2) * self.l1 * self.gravity * cos(self.jointAngle[0]) + self.l2 * self.gravity * self.l2 * cos(self.jointAngle[0] + self.jointAngle[1])
        G21 = self.m2 * self.gravity * self.l2 * cos(self.jointAngle[0] + self.jointAngle[1])
        gravityMatrix = np.array([G11,G21])
        
        self.Inertia_Matrix = Inertia_Matrix
        self.coriolisMatrix = coriolisMatrix
        self.gravityMatrix = gravityMatrix


    '''
          Forward Dynamics
Forward Dynamics is given by “given kinematic and inertial parameters and joint torques and forces as function of time, find the trajectory of manipulator.”

                          Forward  Dynamics of 2 link manipulator
                        𝜽 ̈ = I ^-1 (𝝉 – C(𝜽,𝜽 ̇) −𝑮(𝜽) – f_ext)

I is symmetric positive definite (Mass M is also replaced in place of I)
Computation of 𝜽 ̈ is computationally expensive and requires of ode int(python) or Newton - Euler integration step for finding the parameter values for a period of time.
C(𝜽,𝜽 ̇) - Coriolis Matrix;  𝑮(𝜽) - Gravity matrix; 𝝉 – Torque generated in each joints; f_ext – External
                                                                                            force if applied
    '''
    def fwd_dyn(self,jt):
        p.setRealTimeSimulation(False)
        
        self.jointAngle = jt
        j = self.Do_Jaco(jt)
  ## to make sure the velocity control is off, to enable torque control
        p.setJointMotorControlArray(self.rob_id, self.mvbl_jts,
                                        controlMode = p.VELOCITY_CONTROL,
                                        forces = [0,0])
                                        
                                        
        for i in range(1000):
     
            self.matrices_4_torq(self.jointAngle,self.jointVelocity)
            '''
            Euler Integration algorithm for Forward dynamics
            '''
            self.jointAccel = np.dot( np.linalg.inv(self.Inertia_Matrix) , self.jointTorque -
                            self.coriolisMatrix - self.gravityMatrix - np.dot(j.T,self.ex_F[i]))
            self.jointVelocity = self.jointAccel * self.dt + self.jointVelocity
            self.jointAngle = self.jointVelocity * self.dt + self.jointAngle
            
            print("jt angle",self.jointAngle)
            
            j=self.Do_Jaco(self.jointAngle)
            print(i,"Jacc: ",self.jointAccel)
            self.matrices_4_torq(self.jointAngle,self.jointVelocity)
       
            self.jointTorque = np.dot(self.jointAccel,self.Inertia_Matrix) + self.coriolisMatrix + self.gravityMatrix + np.dot(j.T,self.ex_F[i])
   
            print(i,"Jtorq: ",self.jointTorque)
            y=self.jointTorque
  
            p.setJointMotorControlArray(self.rob_id, self.mvbl_jts,
                                        controlMode = p.TORQUE_CONTROL,
                                        forces = y )

            p.stepSimulation()
            time.sleep(self.dt)
        p.disconnect()

            

        
        
        ### calling my def and other computations
if __name__ == '__main__':

    robo = two_dof_rr()
    robo.createWorld()
    
    jointAngle= np.array([0,0])
    robo.fwd_dyn(jointAngle)
    
            
   

   
