import pygame
import math
import numpy as np

from InvertedPendulumClass import InvertedPendulum

pygame.init()


screen = pygame.display.set_mode((0,0),pygame.FULLSCREEN)

w, h = pygame.display.get_surface().get_size()

pendulumSpace=np.array([1200,400])
pendulumSpace_x_offset=-175
pendulumSpace_center=np.array([w/2+pendulumSpace_x_offset,(h/4)])
pendulumSpace_center2=np.array([w/2+pendulumSpace_x_offset,(h/4*3)])

railSpace=np.array([800,10])

widgetSpace=np.array([300,275])
widgetSpace_x_offset=625
widget1_y_offset=-60
widget2_y_offset=250

widgetSpace_center1_1=np.array([w/2-widgetSpace[0]/2+widgetSpace_x_offset, (h/4)-widgetSpace[1]/2+widget1_y_offset])
widgetSpace_center1_2=np.array([w/2-widgetSpace[0]/2+widgetSpace_x_offset, (h/4)-widgetSpace[1]/2+widget2_y_offset])
widgetSpace_center2_1=np.array([w/2-widgetSpace[0]/2+widgetSpace_x_offset, (h/4*3)-widgetSpace[1]/2+widget1_y_offset])
widgetSpace_center2_2=np.array([w/2-widgetSpace[0]/2+widgetSpace_x_offset, (h/4*3)-widgetSpace[1]/2+widget2_y_offset])


rect = pygame.Rect(pendulumSpace_center[0]-pendulumSpace[0]/2, pendulumSpace_center[1]-pendulumSpace[1]/2, pendulumSpace[0], pendulumSpace[1])
rect2 = pygame.Rect(pendulumSpace_center2[0]-pendulumSpace[0]/2, pendulumSpace_center2[1]-pendulumSpace[1]/2, pendulumSpace[0], pendulumSpace[1])

rail1=pygame.Rect(pendulumSpace_center[0]-railSpace[0]/2, pendulumSpace_center[1]-railSpace[1]/2, railSpace[0], railSpace[1])
rail2=pygame.Rect(pendulumSpace_center2[0]-railSpace[0]/2, pendulumSpace_center2[1]-railSpace[1]/2, railSpace[0], railSpace[1])


rect3 = pygame.Rect(widgetSpace_center1_1[0],widgetSpace_center1_1[1], widgetSpace[0], widgetSpace[1])
rect4 = pygame.Rect(widgetSpace_center1_2[0],widgetSpace_center1_2[1], widgetSpace[0], widgetSpace[1]/3)

rect5 = pygame.Rect(widgetSpace_center2_1[0],widgetSpace_center2_1[1], widgetSpace[0], widgetSpace[1])
rect6 = pygame.Rect(widgetSpace_center2_2[0],widgetSpace_center2_2[1], widgetSpace[0], widgetSpace[1]/3)

cartSize=np.array([30,20])
cart1Pos=np.array([pendulumSpace_center[0],pendulumSpace_center[1]])
cart2Pos=np.array([pendulumSpace_center2[0]+0,pendulumSpace_center2[1]])

cart1=pygame.Rect(cart1Pos[0]-cartSize[0]/2, cart1Pos[1]-cartSize[1]/2, cartSize[0], cartSize[1])
cart2=pygame.Rect(cart2Pos[0]-cartSize[0]/2, cart2Pos[1]-cartSize[1]/2, cartSize[0], cartSize[1])

pendulumSize=20
pendulumLength=150

pendulum1Angle=math.pi
pendulum1Pos=np.array([cart1Pos[0]+math.sin(pendulum1Angle)*pendulumLength,cart1Pos[1]-math.cos(pendulum1Angle)*pendulumLength])

pendulum2Angle=math.pi
pendulum2Pos=np.array([cart2Pos[0]+math.sin(pendulum2Angle)*pendulumLength,cart2Pos[1]-math.cos(pendulum2Angle)*pendulumLength])

font1 = pygame.font.SysFont('mono', 20,False, False)

# ---------------- PHYSICS PARAMETERS ----------------
mc = 3
mp = 0.15
L = pendulumLength / 100  # convert pixels → meters
g = 5
bp = 0.0  # pendulum air resistance

fps=240
dt = 1/240


# ---------------- STATE VARIABLES ----------------
x = 0.0
x_dot = 0.0

theta = pendulum1Angle
theta_dot = 0.0

maxCartVel= 8
force=80
force_player=80
torque=0.5

pend1=InvertedPendulum(cart1Pos[0],cart1Pos[1],pendulum1Angle,mc,mp,L,g,bp,fps)
pend1.setForce(force)
pend2=InvertedPendulum(cart2Pos[0],cart2Pos[1],pendulum2Angle,mc,mp,L,g,bp,fps)
pend2.setForce(force_player)


import torch
import torch.nn as nn
import torch.nn.functional as F

n_observations = 5   # IMPORTANT: must match training
n_actions = 2

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class DQN(nn.Module):

  def __init__(self, n_observations, n_actions):
    super(DQN, self).__init__()
    self.layer1 = nn.Linear(n_observations, 64)
    self.layer2 = nn.Linear(64, 64)
    self.layer3 = nn.Linear(64, n_actions)

  # Called with either one element to determine next action, or a batch
  # during optimization. Returns tensor([[left0exp,right0exp]...]).
  def forward(self, x):
    x = F.tanh(self.layer1(x))
    x = F.tanh(self.layer2(x))
    return self.layer3(x)


policy_net = DQN(n_observations, n_actions).to(device)
policy_net.load_state_dict(torch.load("dqn_pendulum.pth", map_location=device))

run=True


while run:
  screen.fill((120,120,120,0.1))
  pygame.draw.rect(screen, 20, rect,border_radius=20)

  pygame.draw.rect(screen, 20, rect2,border_radius=20)

  pygame.draw.rect(screen, 200, rail1,border_radius=10)

  pygame.draw.rect(screen, 20, rect2,border_radius=20)

  pygame.draw.rect(screen, 200, rail2,border_radius=10)

  pygame.draw.rect(screen, 100, rect3,border_radius=20)
  pygame.draw.rect(screen, 100, rect5,border_radius=20)

  pygame.draw.rect(screen, 150, rect4,border_radius=20)
  pygame.draw.rect(screen, 150, rect6,border_radius=20)
  
  #pygame.draw.rect(screen, 250, cart1,border_radius=5)
  #pygame.draw.rect(screen, 250, cart2,border_radius=5)

  for i in range(41):
    pygame.draw.line(screen, [175,175,175], (pendulumSpace_center[0]-400+i*20,pendulumSpace_center[1]+35), (pendulumSpace_center[0]-400+i*20,pendulumSpace_center[1]+45), 2)

  for i in range(9):
    pygame.draw.line(screen, [250,100,100], (pendulumSpace_center[0]-400+i*100,pendulumSpace_center[1]+30), (pendulumSpace_center[0]-400+i*100,pendulumSpace_center[1]+50), 2)
    textImg = font1.render(str(-400+i*100), True, (255,255,255))
    text_width, text_height = font1.size(str(-400+i*100)) #txt being whatever str you're rendering
    screen.blit(textImg, (pendulumSpace_center[0]-400+i*100-text_width/3,pendulumSpace_center[1]+60))

  for i in range(41):
    pygame.draw.line(screen, [175,175,175], (pendulumSpace_center2[0]-400+i*20,pendulumSpace_center2[1]+35), (pendulumSpace_center2[0]-400+i*20,pendulumSpace_center2[1]+45), 2)

  for i in range(9):
    pygame.draw.line(screen, [250,100,100], (pendulumSpace_center2[0]-400+i*100,pendulumSpace_center2[1]+30), (pendulumSpace_center2[0]-400+i*100,pendulumSpace_center2[1]+50), 2)
    textImg = font1.render(str(-400+i*100), True, (255,255,255))
    text_width, text_height = font1.size(str(-400+i*100)) #txt being whatever str you're rendering
    screen.blit(textImg, (pendulumSpace_center2[0]-400+i*100-text_width/3,pendulumSpace_center2[1]+60))

  
  #pygame.draw.line(screen, [0,255,0], cart1Pos, pendulum1Pos, 4)
  #pygame.draw.line(screen, [0,255,0], cart2Pos, pendulum2Pos, 4)

  #pygame.draw.circle(screen,(255,0,0),pendulum1Pos,pendulumSize)
  #pygame.draw.circle(screen,(255,0,0),pendulum2Pos,pendulumSize)


  key= pygame.key.get_pressed()

  if key[pygame.K_r]:
    pend1.reset()
    pend2.reset()
  

  # ---------------- CONTROL FORCE ----------------
  
  if key[pygame.K_a]:
    pend1.addForceOnPendulum(-torque)
  if key[pygame.K_d]:
    pend1.addForceOnPendulum(torque)

  if key[pygame.K_LEFT]:
    pend2.addForce(-force_player)
  if key[pygame.K_RIGHT]:
    pend2.addForce(force_player)

  state_np = np.array([
    pend1.x,
    pend1.x_dot,
    math.cos(pend1.theta),
    math.sin(pend1.theta),
    pend1.theta_dot
  ], dtype=np.float32)

  state = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)


  with torch.no_grad():
    action = policy_net(state).max(1).indices.item()

  if action == 0:
    pend1.addForce(-force)
  else:
    pend1.addForce(force)
  
  
  pend1.update()
  pend2.update()
  

  #print(x)

  
    

  # ---------------- UPDATE POSITIONS ----------------
  cart1Pos[0] = pendulumSpace_center[0] + pend1.getX() * 100
  cart1.center = cart1Pos
  cart2Pos[0] = pendulumSpace_center2[0] + pend2.getX() * 100
  cart2.center = cart2Pos
  

  pendulum1Pos = pend1.getPendulumPos()
  pendulum2Pos = pend2.getPendulumPos()

  pend1.draw(screen,cart1,cart1Pos)
  pend2.draw(screen,cart2,cart2Pos)

  if key[pygame.K_ESCAPE]:
    run = False
  
  for event in pygame.event.get():
    if event.type==pygame.QUIT:
      run = False
  pygame.display.update()


pygame.quit()