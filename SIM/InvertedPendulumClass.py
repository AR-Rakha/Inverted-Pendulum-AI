import numpy as np
import math
import pygame

class InvertedPendulum:
  def __init__(self, x_pixel, y_pixel, angle, massCart, massPendulum, length_m, gravity, airResistance, fps):
    # Physics state (in meters)
    self.x = 0.0  # displacement in meters
    self.x_dot = 0.0
    self.x_ddot = 0.0

    self.theta = angle
    self.theta_dot = 0.0
    self.theta_ddot = 0.0

    self.mc = massCart
    self.mp = massPendulum
    self.l = length_m  # in meters
    self.g = gravity
    self.frictionP = airResistance
    self.dt = 1/fps
    self.fps=fps

    self.F = 0.0
    self.force_mag=0
    self.torque_ext = 0.0  # external torque applied to pendulum

    self.rail_limit = 4
    self.k_wall = 10000
    self.d_wall = 500

    self.maxCartVel = 8 

    # Pixel offset for drawing (center of pendulum space)
    self.pixelOffset = x_pixel
    self.y_pixel = y_pixel

    # Pendulum length in pixels for drawing
    self.pendulumLengthPix = 150


    self.reward=0
    self.tsReward=0 # reward this TimeStep (ts)
    self.rewardFunctions=[self.rewardFunction1,self.rewardFunction2,self.rewardFunction3]

    self.timeLimit=0 # seconds
    self.time=0

  # ---------------- GETTERS ----------------
  def getX(self):
    return self.x  # small meter displacement; main code multiplies by 100

  def getY(self):
    return self.y_pixel

  def getL(self):
      return self.pendulumLengthPix

  def getAngle(self):
    return self.theta

  def getPendulumPos(self):
    # return pixel coordinates for Pygame drawing
    return np.array([
        self.pixelOffset + self.x * 100 + math.sin(self.theta) * self.pendulumLengthPix,
        self.y_pixel - math.cos(self.theta) * self.pendulumLengthPix
    ])

  def setMaxCartVel(self,maxCartVel):
    self.maxCartVel=maxCartVel

  def setTimeLimit(self,limitInSeconds):
    self.timeLimit=limitInSeconds
  
  def setForce(self,force):
    self.force_mag=force

  def timeReset(self):
    if self.time/self.fps>=self.timeLimit:
      self.reset(math.pi)
      self.resetReward()

  def reset(self,startangle=math.pi):
    self.x=0
    self.x_dot=0
    self.theta=startangle
    self.theta_dot=0

    self.time=0
    self.resetReward()

    return np.array([self.x, self.x_dot, math.cos(self.theta),math.sin(self.theta), self.theta_dot], dtype=np.float32)


  def rewardFunction1(self):
    return (math.cos(self.theta) + 1) / 2
  
  def rewardFunction2(self):
    return (math.cos(self.theta) + 1) / 4 + (-abs(self.x)+self.rail_limit)/(self.rail_limit)
  
  def rewardFunction3(self):
    return (math.cos(self.theta) + 1) / 2 + ((-abs(self.x)+self.rail_limit-2)/(self.rail_limit-2)) - 0.01*self.x_dot**2
  def addReward(self,rewardFunctionNum):
    r=self.rewardFunctions[rewardFunctionNum]()
    self.reward+=r
    self.tsReward=r
      
  
  def addTime(self):
    self.time+=1

  def resetReward(self):
    self.reward=0

  def displayTime(self,screen,textX,textY,fontReward):
    textImg = fontReward.render("Time: "+str(round(self.time/self.fps,2)), True, (255,255,255))
    screen.blit(textImg,[textX,textY])

  def displayReward(self,screen,textX,textY,fontReward):
    textImg = fontReward.render("Reward: "+str(round(self.reward,3)), True, (255,255,255))
    screen.blit(textImg,[textX,textY])
    
  def displayTSReward(self,screen,textX,textY,fontReward):
    textImg = fontReward.render("Reward from action: "+str(round(self.tsReward,3)), True, (255,255,255))
    screen.blit(textImg,[textX,textY])

  def draw(self,screen,cart,cartPos):
    pygame.draw.rect(screen, 250, cart,border_radius=5)
    pygame.draw.line(screen, [0,255,0], cartPos, self.getPendulumPos(), 4)
    pygame.draw.circle(screen,(255,0,0),self.getPendulumPos(),20)

  # ---------------- FORCE ----------------
  def addForce(self, force):
    self.F += force
  
  def addForceOnPendulum(self,torque):
    self.torque_ext+=torque

  # ---------------- UPDATE PHYSICS ----------------
  def update(self):
    # Rail limits
    if self.x > self.rail_limit:
      self.F += -self.k_wall * (self.x - self.rail_limit) - self.d_wall * self.x_dot
    if self.x < -self.rail_limit:
      self.F += -self.k_wall * (self.x + self.rail_limit) - self.d_wall * self.x_dot

    # Equations of motion
    D = self.mc + self.mp - self.mp * (math.cos(self.theta)**2)

    self.x_ddot = (self.F
                    + self.mp * self.l * self.theta_dot**2 * math.sin(self.theta)
                    - self.mp * self.g * math.sin(self.theta) * math.cos(self.theta)
                    + (self.frictionP / self.l) * self.theta_dot * math.cos(self.theta)
                    ) / D

    self.theta_ddot = ((self.g * math.sin(self.theta) - math.cos(self.theta) * self.x_ddot) / self.l
                        - (self.frictionP / (self.mp * self.l**2)) * self.theta_dot
                        + self.torque_ext / (self.mp * self.l**2)
                        )

    # Integration
    self.x_dot += self.x_ddot * self.dt
    self.x_dot = max(min(self.x_dot, self.maxCartVel), -self.maxCartVel)
    self.x += self.x_dot * self.dt

    self.theta_dot += self.theta_ddot * self.dt
    self.theta += self.theta_dot * self.dt

    

    # Reset force after update
    self.F = 0.0
    self.torque_ext = 0.0

  def normalizeTheta(self):
    self.theta = ((self.theta + math.pi) % (2 * math.pi)) - math.pi

  def step(self,action):
    # 0 = left, 1 = right
    force=-self.force_mag if action == 0 else self.force_mag
    self.addForce(force)
    self.update()
    self.normalizeTheta()
    self.addTime()
    self.addReward(2)


    terminated = abs(self.x) > self.rail_limit
    truncated = self.time/self.fps >= self.timeLimit

    # Observation can be [x, x_dot, theta, theta_dot] like CartPole
    # [x, x_dot, cos(theta), sin(theta), theta_dot] is better
    obs = np.array([self.x, self.x_dot, math.cos(self.theta),math.sin(self.theta), self.theta_dot], dtype=np.float32)

    # obs = np.array([self.x,self.x_dot,math.cos(self.theta),math.sin(self.theta),self.theta_dot],dtype=np.float32)
    
    reward = self.tsReward

    info = {}
    return obs, reward, terminated, truncated, info
  
  def render(self,screen,cart_rect,cart_pos,font,rewardX,rewardY,timeX,timeY):
    self.draw(screen, cart_rect, cart_pos)
    self.displayReward(screen, rewardX, rewardY, font)
    self.displayTime(screen, timeX, timeY, font)
