import pygame
import math
import numpy as np

from InvertedPendulumClass import InvertedPendulum

pygame.init()


screen = pygame.display.set_mode((0,0),pygame.FULLSCREEN)

w, h = pygame.display.get_surface().get_size()

pendulumSpace=np.array([1200,800])
pendulumSpace_x_offset=-175
pendulumSpace_center=np.array([w/2+pendulumSpace_x_offset,(h/2)])

railSpace=np.array([800,10])

widgetSpace=np.array([300,235])
widgetSpace_x_offset=615
widget1_y_offset=-60
widget2_y_offset=250

widgetSpace_center1_1=np.array([w/2-widgetSpace[0]/2+widgetSpace_x_offset, (h/4)-widgetSpace[1]/2+widget1_y_offset])
widgetSpace_center1_2=np.array([w/2-widgetSpace[0]/2+widgetSpace_x_offset, (h/4*2.25)-widgetSpace[1]/2+widget1_y_offset])
widgetSpace_center1_3=np.array([w/2-widgetSpace[0]/2+widgetSpace_x_offset, (h/4*3.5)-widgetSpace[1]/2+widget1_y_offset])
widgetSpace_center2_1=np.array([w/2-widgetSpace[0]/2+widgetSpace_x_offset, (h/4*3)-widgetSpace[1]/2+widget1_y_offset])
widgetSpace_center2_2=np.array([w/2-widgetSpace[0]/2+widgetSpace_x_offset, (h/4*3)-widgetSpace[1]/2+widget2_y_offset])


rect = pygame.Rect(pendulumSpace_center[0]-pendulumSpace[0]/2, pendulumSpace_center[1]-pendulumSpace[1]/2, pendulumSpace[0], pendulumSpace[1])

rail1=pygame.Rect(pendulumSpace_center[0]-railSpace[0]/2, pendulumSpace_center[1]-railSpace[1]/2, railSpace[0], railSpace[1])


rect3 = pygame.Rect(widgetSpace_center1_1[0],widgetSpace_center1_1[1], widgetSpace[0], widgetSpace[1])
rect4 = pygame.Rect(widgetSpace_center1_2[0],widgetSpace_center1_2[1], widgetSpace[0], widgetSpace[1])

rect5 = pygame.Rect(widgetSpace_center1_3[0],widgetSpace_center1_3[1], widgetSpace[0], widgetSpace[1])
rect6 = pygame.Rect(widgetSpace_center2_2[0],widgetSpace_center2_2[1], widgetSpace[0], widgetSpace[1]/3)

cartSize=np.array([30,20])
cart1Pos=np.array([pendulumSpace_center[0],pendulumSpace_center[1]])

cart1=pygame.Rect(cart1Pos[0]-cartSize[0]/2, cart1Pos[1]-cartSize[1]/2, cartSize[0], cartSize[1])

pendulumSize=20
pendulumLength=150

pendulum1Angle=math.pi
pendulum1Pos=np.array([cart1Pos[0]+math.sin(pendulum1Angle)*pendulumLength,cart1Pos[1]-math.cos(pendulum1Angle)*pendulumLength])

font1 = pygame.font.SysFont('Mono', 20,False, False)

# ---------------- PHYSICS PARAMETERS ----------------
mc = 3
mp = 0.15
L = pendulumLength / 100  # convert pixels → meters
g = 5
bp = 0.0  # pendulum air resistance

fps=240
dt = 1/fps

# ---------------- STATE VARIABLES ----------------

force=30

pend1=InvertedPendulum(cart1Pos[0],cart1Pos[1],pendulum1Angle,mc,mp,L,g,bp,fps)
pend1.setTimeLimit(20)
pend1.setForce(force)

run=True

while run:  
  pend1.timeReset()
  screen.fill((120,120,120,0.1))
  pygame.draw.rect(screen, 20, rect,border_radius=20)

  pygame.draw.rect(screen, 200, rail1,border_radius=10)



  pygame.draw.rect(screen, 100, rect3,border_radius=20)
  pygame.draw.rect(screen, 100, rect5,border_radius=20)
  pygame.draw.rect(screen, 150, rect4,border_radius=20)
  
  

  for i in range(41):
    pygame.draw.line(screen, [175,175,175], (pendulumSpace_center[0]-400+i*20,pendulumSpace_center[1]+35), (pendulumSpace_center[0]-400+i*20,pendulumSpace_center[1]+45), 2)

  for i in range(9):
    pygame.draw.line(screen, [250,100,100], (pendulumSpace_center[0]-400+i*100,pendulumSpace_center[1]+30), (pendulumSpace_center[0]-400+i*100,pendulumSpace_center[1]+50), 2)
    textImg = font1.render(str(-400+i*100), True, (255,255,255))
    text_width, text_height = font1.size(str(-400+i*100)) #txt being whatever str you're rendering
    screen.blit(textImg, (pendulumSpace_center[0]-400+i*100-text_width/3,pendulumSpace_center[1]+60))

  key= pygame.key.get_pressed()

  # ---------------- CONTROL FORCE ----------------
  F = 0
  if key[pygame.K_LEFT]:
    F = -force
    pend1.addForce(-force)
  if key[pygame.K_RIGHT]:
    F = force
    pend1.addForce(force)

  if key[pygame.K_r]:
    pend1.reset(0)

  if key[pygame.K_e]:
    pend1.reset(math.pi)
  
  # ---------------- RAIL WALL FORCE (BEFORE PHYSICS) ----------------
  pend1.update()
  pend1.addTime()
  pend1.addReward(2)


  #print(x)

  
    

  # ---------------- UPDATE POSITIONS ----------------
  cart1Pos[0] = pendulumSpace_center[0] + pend1.getX() * 100
  cart1.center = cart1Pos
  

  pendulum1Pos = pend1.getPendulumPos()

  pend1.draw(screen,cart1,cart1Pos)

  if key[pygame.K_ESCAPE]:
    run = False
  
  for event in pygame.event.get():
    if event.type==pygame.QUIT:
      run = False

  pend1.displayReward(screen,200,200,font1)
  pend1.displayTSReward(screen,200,230,font1)
  pend1.displayTime(screen,200,300,font1)
  pygame.display.update()


pygame.quit()