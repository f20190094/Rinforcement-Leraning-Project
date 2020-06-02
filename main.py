import os
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']=""

# importing dependencies

import random
import gym
import numpy as np
from _collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os




import pygame as py
import random
import test
import numpy as np

from pygame.sprite import Group

vec=py.math.Vector2
wwidth=485
wheight=640
FPS=60
gravity = 0.5
pipe_vel=3
up_vel = 16

#colors
white=(255,255,255)
black=(0,0,0)
red = (45,205,7)

class Flappy:
    pipesdown: Group

    def __init__(self):
        py.init()
        py.mixer.init()
        self.bases = py.sprite.Group()
        self.tops = py.sprite.Group()
        self.pipesup = py.sprite.Group()
        self.pipesdown = py.sprite.Group()
        self.obstacles = py.sprite.Group()
        self.player = Player()
        self.bg = py.image.load("background.png")
        self.bg = py.transform.scale(self.bg, (1280, 720))
        self.win = py.display.set_mode((wwidth, wheight))
        py.display.set_caption("Flappy Bird")
        self.clock = py.time.Clock()
        self.all_sprites = py.sprite.Group()
        self.running = True
        self.is_done =  False
   # def reset(self):
        self.base1 = Base(0,557)
        self.base2 = Base(wwidth-1, 557)
        self.top1 = Top(0, -1)
        self.top2 = Top(wwidth - 1, -1)
        self.pipe1 = Pipe1(300,300)
        self.pipe2 = Pipe2(300,300)
        self.y = random.randint(100,540)
        self.npipe1 = Pipe1(wwidth + 150, self.y)
        self.npipe2 = Pipe2(wwidth + 150, self.y)
        self.obstacles.add(self.pipe1)
        self.obstacles.add(self.pipe2)
        self.obstacles.add(self.base1)
        self.obstacles.add(self.base2)
        self.obstacles.add(self.top1)
        self.obstacles.add(self.top2)
        self.all_sprites.add(self.player)
        self.all_sprites.add(self.base1)
        self.all_sprites.add(self.top1)
        self.all_sprites.add(self.base2)
        self.all_sprites.add(self.top2)
        self.all_sprites.add(self.pipe1)
        self.all_sprites.add(self.pipe2)
        self.all_sprites.add(self.npipe1)
        self.all_sprites.add(self.npipe2)
        self.pipesdown.add(self.pipe1)
        self.pipesup.add(self.pipe2)
        self.pipesdown.add(self.npipe1)
        self.pipesup.add(self.npipe2)
        self.bases.add(self.base1)
        self.bases.add(self.base2)
        self.tops.add(self.top1)
        self.tops.add(self.top2)
        self.playing = True
        self.activate = False

    def run(self):

        while self.playing:
            self.clock.tick(FPS)
            self.events()
            self.update()
            self.draw()

    def evaluate(self):
        p = self.get_pipeup()
        y = p.rect.x
        if self.player.rect.x == p.rect.x + 52:
            return 30
        if self.is_done:
            return -10
        return 1

    def observe(self):
        pd = self.get_pipedown()
        pu = self.get_pipeup()
        x = pu.x
        return np.array([pd.y - self.player.rect.y, pu.y - self.player.rect.y,self.player.rect.y, wheight - self.player.rect.y, pu.rect.x - self.player.rect.x])

    def update(self):

        hits =  py.sprite.spritecollide(self.player,self.obstacles,False)
        if hits:
            #self.playing = False
            self.is_done = True
            print("hi")

            '''self.player.pos.y = hits[0].rect.top
            self.player.v.y=0'''

        #hits2 = py.sprite.spritecollide(self.player,self.obstacles,False)

        for pipe in self.pipesup:
            if pipe.rect.x + 80 < 0:

                pipe.kill()
                self.spwan()
        for pipe in self.pipesdown:
            if pipe.rect.x + 80 < 0:

                pipe.kill()

        for base in self.bases:
            if base.rect.x + wwidth <= 0:
                base.kill()
                b = Base(wwidth-5,557)
                self.bases.add(b)
                self.obstacles.add(b)
                self.all_sprites.add(b)

        for top in self.tops:
            if top.rect.x + wwidth <= 0:
                top.kill()
                b = Top(wwidth-5,0)
                self.tops.add(b)
                self.obstacles.add(b)
                self.all_sprites.add(b)

        self.all_sprites.update()

    def get_pipeup(self):
        max_diff = 1000

        for pipe in self.pipesup:

            if abs(pipe.rect.x + 80 - self.player.rect.x) < max_diff:
                max_diff = abs(pipe.rect.x + 80 - self.player.rect.x)
                p = pipe
        return p

    def get_pipedown(self):
        max_diff = 1000

        v=self.pipesdown
        #plist = [self.pipesdown (i) for i in range(len(self.pipesdown))]
        for pipe in self.pipesdown:

            if abs(pipe.rect.x + 80 - self.player.rect.x) < max_diff:

                p = pipe
                max_diff = abs(pipe.rect.x + 80 - self.player.rect.x)

        return p

    def action(self, action):
        if action == 1:
            self.player.v.y += -up_vel
            if self.player.v.y < 0:
                self.player.v.y = max(-9, self.player.v.y)

    def spwan(self):
        y = generator()#300#random.randint(100,540)
        d = wwidth + 150
        p1 = Pipe1(d, y)
        p2 = Pipe2(d, p1.y)
        self.all_sprites.add(p1)
        self.all_sprites.add(p2)
        self.obstacles.add(p1)
        self.obstacles.add(p2)
        self.pipesdown.add(p1)
        self.pipesup.add(p2)

    def events(self):
        for event in py.event.get():
            if event.type == py.QUIT:
                if self.playing:
                    self.playing=False
                self.running = False
            if event.type == py.KEYDOWN:
                if event.key == py.K_SPACE:
                    if not self.player.pressed:
                        self.player.v.y += -up_vel
                        if self.player.v.y < 0:
                            self.player.v.y = max(-9, self.player.v.y)
                        self.player.pressed=True

            if event.type == py.KEYUP:
                if event.key == py.K_SPACE:
                    self.player.pressed = False

    def view(self):
        self.win.fill(white)
        self.image = py.Surface((288, 512))
        self.image.blit(self.bg, (0, 0))
        self.image = py.transform.scale(self.image, (wwidth, wheight))
        #self.image.set_colorkey(black)
        self.win.blit(self.bg, (0,0))
        #self.win = py.transform.scale(self.win, (wwidth, wheight))
        self.all_sprites.draw(self.win)
        self.bases.draw(self.win)
        self.tops.draw(self.win)
        py.display.flip()

    def show_start_screen(self):
        pass

    def show_go_screen(self):
        pass

class Player(py.sprite.Sprite):
    def __init__(self):
        py.sprite.Sprite.__init__(self)
        self.bird=py.image.load("bird1.png")
        self.image=py.Surface((34,24))
        self.image.blit(self.bird,(0,0))
        self.image = py.transform.scale(self.image, (58, 41))
        self.image.set_colorkey(black)
        self.rect=self.image.get_rect()
        self.rect.topleft=(wwidth/4,0)
        self.pos=vec(wwidth/4,wheight*0.3)
        self.v=vec(0,0)
        self.a=vec(0,0)
        self.pressed = False

    def update(self):
        self.a = vec(0,gravity)

        for event in py.event.get():
            if event.type == py.KEYDOWN:
                if event.key == py.K_SPACE:
                    if not self.pressed:
                        self.v.y += -up_vel
                        self.pressed=True

            if event.type == py.KEYUP:
                if event.type == py.K_SPACE:
                    self.pressed = False

        '''keys=py.key.get_pressed()
        if keys[py.K_SPACE]:
            self.v.y += -3'''
        self.v += self.a
        self.pos += self.v + 0.5*self.a
        if self.pos.y > wheight:
            self.a = vec(0,0)
            self.v = vec(0,0)
            if self.v.y  < 0:
                self.v.y = max(-3,self.v.y)
            self.pos.y = wheight
        self.rect.midbottom = self.pos

class Base(py.sprite.Sprite):
    def __init__(self,x,y):
        py.sprite.Sprite.__init__(self)
        self.b = py.image.load("base.png")
        self.x = x
        self.y = y
        self.image = py.Surface((336,112))
        self.image.blit(self.b,(0,0))
        self.image = py.transform.scale(self.image, (wwidth,wheight - y))
        self.image.set_colorkey(black)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
    def update(self):
        self.rect.x += -pipe_vel

class Top(py.sprite.Sprite):
    def __init__(self,x,y):
        py.sprite.Sprite.__init__(self)
        self.x = x
        self.y = y
        self.image = py.Surface((wwidth,1))
        self.image.fill(red)
        #self.image.blit(self.b,(0,0))
        #self.image = py.transform.scale(self.image, (wwidth,wheight - y))
        #self.image.set_colorkey(black)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
    def update(self):
        self.rect.x += -pipe_vel

class Pipe1(py.sprite.Sprite):
    def __init__(self, x, y):
        py.sprite.Sprite.__init__(self)
        self.p = py.image.load("pipe.png")
        self.x = x
        self.y = y
        '''
        self.image = py.Surface((34, 24))
        self.image.blit(self.bird, (0, 0))
        self.image = py.transform.scale(self.image, (58, 41))
        self.image.set_colorkey(black)
        '''
        self.image = py.Surface((52,320))
        self.image.blit(self.p,(0,0))
        self.image = py.transform.scale(self.image, (80, 492))
        self.image.set_colorkey(black)
        #self.image.fill(red)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.vel = pipe_vel

    def update(self):
        self.rect.x += -self.vel

class Pipe2(py.sprite.Sprite):
    def __init__(self, x, y):
        py.sprite.Sprite.__init__(self)

        #self.PIPE_TOP = pygame.transform.flip(pipe_img, False, True)
        self.p = py.image.load("pipe.png")
        self.x = x
        self.y = y - 145
        self.image = py.Surface((52, 320))
        self.image.blit(self.p, (0, 0))
        self.image = py.transform.scale(self.image, (80, 492))
        self.image.set_colorkey(black)
        self.image = py.transform.flip(self.image, False, True)
        self.rect = self.image.get_rect()
        self.rect.bottomleft = (x,y-175)
        self.vel = pipe_vel

    def update(self):
        self.rect.x += -self.vel



def generator():
    return random.randint(180, 540)





class CustomEnv():

    def __init__(self):
        self.pygame = Flappy()
        #self.acition_space = 2
        #self.observation_space = spaces.Box()

    def reset(self):
        del self.pygame
        self.pygame = Flappy()
        obs = self.pygame.observe()
        return obs                                #done       return distances from pipes, below, above, below pipe, above pipe

    def step(self, action):
        self.pygame.action(action)                #done       take action according to given action
        self.pygame.update()
        obs = self.pygame.observe()               #done       return distances from pipes, below, above, below pipe, above pipe
        reward = self.pygame.evaluate()           #done       return 30 if passes and 1 for floating
        done = self.pygame.is_done                #done       return true if playing == False
        return obs,reward,done,{}

    def render(self):
        self.pygame.view()





# set parameterrs

env = CustomEnv()
state_size = 5
action_size = 2
n_episodes = 20000
output_dir = 'model_output/cartpole'
batch_size = 32

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

### modelbuilding

def DQNmodel(state_size,action_size,learning_rate):
    model = Sequential()
    model.add(Dense(24, input_dim=state_size, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))

    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))

    return model

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size=state_size
        self.action_size=action_size
        self.memory=deque(maxlen=2000)
        self.gamma=0.99
        self.epsilon =1.0
        self.epsilon_decay=0.999
        self.epsilon_min=0.01
        self.learning_rate=0.001
        self.model=DQNmodel(self.state_size,self.action_size,self.learning_rate)       #
        self.sample_model=DQNmodel(self.state_size,self.action_size,self.learning_rate)

    def _build_model(self):
        model=Sequential()
        model.add(Dense(24,input_dim=self.state_size,activation='relu'))
        model.add(Dense(24,activation='relu'))
        model.add(Dense(self.action_size,activation='softmax'))

        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))

        return model
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))

    def act(self,state):
        if np.random.rand()<=self.epsilon:
            return random.randrange(self.action_size)
        act_values=self.model.predict(state)        #
        return np.argmax(act_values[0])

    def replay(self,batch_size,episode):
        minibatch=random.sample(self.memory,batch_size)

        for state,action,reward,next_state,done in minibatch:
            target=reward
            if not done:
                target=(reward+self.gamma*np.max(self.sample_model.predict(next_state)[0]))        #
            target_f=self.model.predict(state)
            target_f[0][action]=target

            self.model.fit(state,target_f,epochs=1,verbose=0)
        if episode%10==0:                                                   #
            self.sample_model.set_weights(self.model.get_weights())         #
        if self.epsilon>self.epsilon_min:
            self.epsilon*=self.epsilon_decay
    def load(self,name):
        self.model.load_weights(name)

    def save(self,name):
        self.model.save_weights(name)

agent=DQNAgent(state_size,action_size)

done=False
for e in range(n_episodes):
    state=env.reset()
    state=np.reshape(state,[1,state_size])
    render=False
    #if e>980:
    render=True
    tot_reward=0
    for time in range(500):

        if render:
            env.render()
        action=agent.act(state)
        next_state, reward, done, _ = env.step(action)
        #reward = reward if not done else 100
        next_state=np.reshape(next_state,[1,state_size])
        agent.remember(state,action,reward,next_state,done)

        state=next_state
        tot_reward += reward
        if done:
            print("episode: {}/{}, score: {},ee: {:.2}".format(e,n_episodes,tot_reward,agent.epsilon))
            break


    if len(agent.memory)>batch_size:
        agent.replay(batch_size,e)

    if e%50==0:
        agent.save(output_dir+"weights_"+'{:04d}'.format(e)+".hdf5")

