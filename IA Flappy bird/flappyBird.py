import pygame
import random
import neat
import graphviz


# Initialisation de Pygame
pygame.init()

# Création de la fenêtre de jeu
screen_width = 400
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Flappy Bird")

# Définition des couleurs
white = (255, 255, 255)
black = (0, 0, 0)
yellow = (255,255,0)
red = (255,0,0)


# Définition de la taille de l'oiseau
bird_size = 30

# Définition de la taille des tuyaux
pipe_width = 50
pipe_gap = 200

# Définition de la vitesse de déplacement des tuyaux
pipe_speed = 0.1

# Définition de la gravité
gravity = 0.001


#Dessin neuronne
def draw_net(config, genome, view=False):
    # Création du graphique
    g = graphviz.Digraph(format='png')

    # Ajout des nœuds du réseau
    for i in range(genome.nb_inputs):
        g.node(str(i), color='blue')

    for i in range(genome.nb_outputs):
        g.node(str(i+genome.nb_inputs), color='red')

    for i, node in genome.nodes.items():
        if i < genome.nb_inputs or i > genome.nb_inputs+genome.nb_outputs-1:
            g.node(str(i))

    # Ajout des connexions entre les nœuds
    for cg in genome.connections.values():
        if cg.enabled:
            g.edge(str(cg.key[0]), str(cg.key[1]), label="{:.2f}".format(cg.weight))

    # Affichage ou enregistrement du graphique
    if view:
        g.view()
    else:
        g.render('network', cleanup=True)
        
        
        
# Définition de la classe Bird
class Bird:
    def __init__(self):
        self.x = 50 
        self.y = screen_height/2
        self.vy = 0
        self.size = bird_size
        self.color  = (random.randint(0,255), random.randint(0,255), random.randint(0,255))

    def jump(self):
        self.vy = -0.1 

    def move(self):
        self.y += self.vy
        self.vy += gravity

    def draw(self) : 
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.size, self.size))
        

    def check_boundaries(self):
        if self.y < 0:
            return True
        elif self.y + self.size > screen_height:
             return True
        return False
    
    def check_pipes(self, pipes) : 
         # Vérification si l'oiseau touche un tuyau
        for pipe in pipes:
            if self.x + bird_size > pipe.x and self.x < pipe.x + pipe_width:
                if self.y < pipe.height or self.y + bird_size > pipe.height + pipe_gap:
                    return True
        return False
        

# Définition de la classe Pipe
class Pipe:
    def __init__(self):
        self.x = screen_width
        self.height = random.randint(50, screen_height - pipe_gap - 50)
        self.color = white

    def move(self):
        self.x -= pipe_speed

    def draw(self):
        pygame.draw.rect(screen, self.color, (self.x, 0, pipe_width, self.height))
        pygame.draw.rect(screen, self.color, (self.x, self.height + pipe_gap, pipe_width, screen_height))
        
        




def main(genomes, config) :
    # Initialisation de l'oiseau et des tuyaux
    nets = []
    ge =[]
    birds = []
    
    for _,g in genomes : 
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird())
        g.fitness = 0
        ge.append(g)
        
    
    pipes = [Pipe()]



    
  
    # Boucle de jeu
    while True:
        # Vérification des événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
           


    
        #Deplacement des oiseaux avec IA 
        pipe_ind = 0
        if len(birds) > 0 :
            if(len(pipes) > 0 and birds[0].x > pipes[0].x + pipe_width) : 
                pipe_ind =1
        else :
            break
        
        pipes[pipe_ind].color = red
    
        
                
        for x, bird in enumerate(birds) : 
            ge[x].fitness += 0.1
            
            output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - (pipes[pipe_ind].height + pipe_gap))))
            
            if output[0] >0.5 : 
                bird.jump()

            
        # Déplacement des oiseaus
        for bird in birds : 
            bird.move()
            
      
        
        #RECOMPENSE
        for x,bird in enumerate(birds) :
            if(bird.check_pipes(pipes)) :
                ge[x].fitness -= 1
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)
                
        
        #Si dépasse écran :
        for x,bird in enumerate(birds) : 
            if(bird.check_boundaries()) : 
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)
                
                

        # Déplacement des tuyaux
        for pipe in pipes:
            pipe.move()

        # Vérification si un nouveau tuyau doit être ajouté
        if pipes[-1].x < screen_width - 200:
            for g in ge : 
                g.fitness += 5
            pipes.append(Pipe())

        # Vérification si un tuyau doit être supprimé
        if pipes[0].x < -pipe_width:
            pipes.pop(0)

        
        
        
        
        # Dessin de l'écran
        screen.fill(black)
        
        for bird in birds : 
            bird.draw()
            
        for pipe in pipes:
            pipe.draw()
            
        pygame.display.update()
 
    
    
def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    

    p = neat.Population(config) #Population de base

    p.add_reporter(neat.StdOutReporter(True)) #stats
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    winner = p.run(main,50)




run("./config-feedforward.txt")
    
        

        



