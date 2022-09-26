from tokenize import Triple
from turtle import forward
import pygame
import time
import random
import numpy as np


print("Fin de l'importation")

pygame.init()

TAILLE_ECRAN = 40   
TAILLE_IMAGE = 25
X_OISEAU = TAILLE_ECRAN//2
TAILLE_SAUT = 3
ECRAN = pygame.display.set_mode((TAILLE_ECRAN*TAILLE_IMAGE, TAILLE_ECRAN*TAILLE_IMAGE))
ESPACE_TUYAU = 25
TROU_TUYAU = 4
EPPAISSEUR_TUYAU = 3
NBR_OISEAUX = 30
DIMENSION_RESEAU = [3,6,1]
TEMPS_ATTENTE = 0.05
NOMBRE_GENERATION = 20


#Image pour la partie graphique 
LOAD_CIEL = pygame.image.load("ciel.png").convert()
IMAGE_CIEL = pygame.transform.scale(LOAD_CIEL, (TAILLE_IMAGE, TAILLE_IMAGE))

LOAD_TUYAU = pygame.image.load("tuyau.png").convert()
IMAGE_TUYAU = pygame.transform.scale(LOAD_TUYAU, (TAILLE_IMAGE, TAILLE_IMAGE))

IMAGE_OISEAUX = [None]*NBR_OISEAUX
for i in range(NBR_OISEAUX):
    if i < 15:
        nom_oiseau = 'oiseau'+str(i+1)+'.png'
    else:
        nom_oiseau = 'oiseau'+str(i-14)+'.png'

    LOAD_OISEAU =  pygame.image.load(nom_oiseau).convert()
    IMAGE_OISEAUX[i] = pygame.transform.scale(LOAD_OISEAU, (TAILLE_IMAGE, TAILLE_IMAGE))



class Reseau_neurone():

    def __init__(self, param):

        self.param = param 

    def forward_propagation(self, etat):
        reshape_etat = [[etat[0]], [etat[1]], [etat[2]]]
        activations = {'A0' : reshape_etat}

        C = len(self.param) // 2
        
        for c in range(1,C+1):
            Z = self.param['W'+str(c)].dot(activations['A'+str(c-1)]) + self.param['b'+str(c)]
            activations['A'+str(c)] = 1 / (1+np.exp(-Z))
        return activations['A2'][0][0] > 0.5


class Game():

    def __init__(self):

        self.tuyau = True 
        self.x_tuyau = random.randint(5,TAILLE_ECRAN-TROU_TUYAU-5)
        self.tab = self.reset_tab()
        self.mouvement = 0
    

    def reset_tab(self):

        '''Ne prend rien en paramètre mais renvoie le tableau de jeu au début '''

        tab =  [[0 for j in range(TAILLE_ECRAN)] for i in range(TAILLE_ECRAN)]
        tab[X_OISEAU][X_OISEAU] = 2
        return tab


    def deplacement_gauche(self):
        
        '''Prend en paramètre le tableau de jeu
            modifie le tableau de jeu
            renvoie rien'''

        
        for i in range(TAILLE_ECRAN):
            place = False
            del self.tab[i][0]

            if self.tuyau:
                if self.x_tuyau > i or i > self.x_tuyau + TROU_TUYAU:
                    self.tab[i].append(1)
                    place = True
            if not place:
                self.tab[i].append(0)
                
        return None
    

class Oiseau():

    def __init__(self, numero, param):
        
        self.mouvement = 0
        self.y_oiseau = X_OISEAU
        self.numero = numero
        self.continuer = True
        self.reseau = Reseau_neurone(param)
        self.etat_final = []
        self.score = 0

    def tomber(self, tab):
        '''Prend en paramètre le tableau et le modifie pour que l'oiseau tombe
        renvoie False si l'oiseau touche le sol ou un tuyau et True sinon'''

        tab[self.y_oiseau][X_OISEAU] = 0
        if self.y_oiseau == TAILLE_ECRAN-1 or tab[self.y_oiseau+1][X_OISEAU] == 1:
            return False
        
        tab[self.y_oiseau+1][X_OISEAU] = self.numero
        self.y_oiseau += 1
        return True
    
    def sauter(self, tab):
        
        '''prend en paramètre le tableau
            modifie le tableau en faisant sauter l'oiseau
            renvoie True si il ne meurt pas et False sinon'''
      
        for _ in range(TAILLE_SAUT):
            tab[self.y_oiseau][X_OISEAU] = 0

            if self.y_oiseau > 0 and tab[self.y_oiseau-1][X_OISEAU] == 0:
                tab[self.y_oiseau-1][X_OISEAU] = self.numero
                self.y_oiseau -= 1
            else:
                return False 
        return True

    def avancer(self, tab):

        '''Prend en paramètre le tableau
        modifie le tableau en faisant avancer l'oiseau
        renvoie False s'il meurt et False sinon'''


        tab[self.y_oiseau][X_OISEAU] = 0

        if tab[self.y_oiseau][X_OISEAU+1] == 1:
            return False
            
        tab[self.y_oiseau][X_OISEAU+1] = self.numero
        return True
        

    def choix_action(self, etat):
        
        saut = self.reseau.forward_propagation(etat)
        if saut:
            return 1
        return 0

    
    def etat_jeu(self, tab, x_tuyau):
        rep = [0,0,0]

        i = X_OISEAU
        while i < TAILLE_ECRAN-1 and tab[0][i] != 1:
            i += 1
        rep[0] = (i-X_OISEAU)/(TAILLE_ECRAN//2)
        
        haut = (self.y_oiseau-x_tuyau)+1
        bas = (x_tuyau+TROU_TUYAU)-self.y_oiseau-1
        bas_haut = [haut, bas]

        for j in range(2):
            if bas_haut[j] > 0:
                rep[j+1] = bas_haut[j]/TAILLE_ECRAN
            else:
                rep[j+1] = 0
        return np.array(rep)



def initialisation_parametre(dimension):

    parametres = {}
    C = len(dimension)

    for c in range(1,C):
        parametres['W' + str(c)] = np.random.randn(dimension[c],dimension[c-1])
        parametres['b' + str(c)] = np.random.randn(dimension[c],1)
    
    return parametres

def affichage(tab, nbr_oiseaux):
        
        '''Ne prend rien en paramètre et ne renvoie rien mais affiche le tableau avec une partie graphique '''
        image = [None]*(nbr_oiseaux+2)
        image[0] = IMAGE_CIEL
        image[1] = IMAGE_TUYAU
        for i in range(nbr_oiseaux):
            image[i+2] = IMAGE_OISEAUX[i]
        
        for i in range(TAILLE_ECRAN):
            for j in range(TAILLE_ECRAN):
                ECRAN.blit(image[tab[i][j]], (j*TAILLE_IMAGE, i*TAILLE_IMAGE))
        pygame.display.flip()
        return None

def fin_de_jeu(tab):
    for oiseau in tab:
        if oiseau.continuer == True:
            return True
    return False 

def supprimer_double(tab):
    rep = []
    for val in tab:
        if not val in rep:
            rep.append(val)
    return rep


def faire_jouer_IA(oiseaux):

    pygame.init()
    jeu = Game()
    mettre_tuyau = 0
    TEMPS_ATTENTE = 0.05
    while fin_de_jeu(oiseaux):
        
        jeu.tuyau = False

        if mettre_tuyau == 0:
            jeu.tuyau = True
            jeu.x_tuyau = random.randint(6, TAILLE_ECRAN-TROU_TUYAU-7)
            mettre_tuyau = ESPACE_TUYAU
            
        if mettre_tuyau > ESPACE_TUYAU-EPPAISSEUR_TUYAU:
            jeu.tuyau = True
        

        for oiseau in oiseaux:
            if oiseau.continuer == True:

                if oiseau.mouvement > 100:
                    TEMPS_ATTENTE = 0.02
                if oiseau.mouvement > 250:
                    TEMPS_ATTENTE = 0.008
                if oiseau.mouvement > 1000:
                    TEMPS_ATTENTE = 0.005
                if oiseau.mouvement > 2000:
                    TEMPS_ATTENTE = 0.003

                oiseau.mouvement += 1
                etat = oiseau.etat_jeu(jeu.tab, jeu.x_tuyau)
                action = oiseau.choix_action(etat)
                if action == 0:
                    oiseau.continuer = oiseau.sauter(jeu.tab)
                else:
                    oiseau.continuer = oiseau.tomber(jeu.tab)
                
                if oiseau.continuer:
                    oiseau.continuer = oiseau.avancer(jeu.tab)

                if oiseau.continuer == False:
                    oiseau.etat_final = etat
        jeu.deplacement_gauche()
        affichage(jeu.tab, NBR_OISEAUX)
        time.sleep(TEMPS_ATTENTE)

        mettre_tuyau -= 1
    

    all_score = {}
    compt = 0
    score_trie = [0]*NBR_OISEAUX
    for oiseau in oiseaux:
        oiseau.score = oiseau.mouvement + (1-(oiseau.etat_final[1]+oiseau.etat_final[2]))
        if oiseau.score in all_score:
            all_score[oiseau.score].append(oiseau)
        else:
            all_score[oiseau.score] = [oiseau]
        score_trie[compt] = oiseau.score
        compt += 1

    score_trie.sort(reverse=True)
    score_trie = supprimer_double(score_trie)

    compt = 0
    reseau_trie = [None]*NBR_OISEAUX
    for s in score_trie:
        for oiseau in all_score[s]:
            reseau_trie[compt] = oiseau.reseau.param
            compt += 1
            
            
    return reseau_trie, score_trie[0]


def melange(reseaux):

    param = initialisation_parametre(DIMENSION_RESEAU)
    for c in param:
        for i in range(len(param[c])):
            for j in range(len(param[c][i])):
                param[c][i][j] = (reseaux[0][c][i][j] + reseaux[1][c][i][j])/2

    return param

def enfant(reseaux):

    param = initialisation_parametre(DIMENSION_RESEAU)
    for c in param:
        for i in range(len(param[c])):
            for j in range(len(param[c][i])):
                param[c][i][j] = (reseaux[0][c][i][j] + reseaux[1][c][i][j])/2

    return param


def aleatoire(param):
    
    for c in param:
        for i in range(len(param[c])):
            for j in range(len(param[c][i])):
                nbr = random.randint(1, 10)
                if nbr == 3:
                    param[c][i][j] = random.randint(-10,10)*0.1

    return param



def mutation(reseaux):

    nouvelle_generation_reseau = [None]*NBR_OISEAUX  
    for i in range(3):
        nouvelle_generation_reseau[i] = reseaux[i]
    
    for i in range(3, 17):
        nouvelle_generation_reseau[i] = aleatoire(melange([reseaux[random.randint(0,3)], reseaux[random.randint(0,3)]]))
    for i in range(17, 30):
        nouvelle_generation_reseau[i] = aleatoire(enfant([reseaux[random.randint(0,3)], reseaux[random.randint(0,3)]]))


    oiseaux = [None]*NBR_OISEAUX
    for i in range(NBR_OISEAUX):
        oiseaux[i] = Oiseau(i+2, nouvelle_generation_reseau[i])

    
    return oiseaux
    

def initialisation_pop():
    
    oiseaux = [None]*NBR_OISEAUX

    for i in range(NBR_OISEAUX):
        oiseaux[i] = Oiseau(i+2, initialisation_parametre(DIMENSION_RESEAU))

    return oiseaux 



def selection_naturel(nbr_generation):

    oiseaux = initialisation_pop()

    for gen in range(1,nbr_generation+1):

        reseaux_trie, meilleur_score = faire_jouer_IA(oiseaux)
        oiseaux = mutation(reseaux_trie)
        print('Generation {}, meilleur score {}'.format(gen, meilleur_score))

        print(reseaux_trie[1])

    return None
    

selection_naturel(NOMBRE_GENERATION)





bon_param = {'W1': np.array([[ 0.0098228 ,  0.14540632,  1.35843347],
       [-0.58004157, -0.59734984,  0.38598107],
       [ 2.40026897,  0.34866012, -1.3264288 ],
       [ 0.1       ,  0.63421583,  0.01941882],
       [ 0.5875295 , -0.75479969, -0.52693874],
       [ 0.8       ,  1.11665393,  0.09572516]]), 'b1': np.array([[-1.13961159],
       [ 0.90227208],
       [ 1.05216931],
       [ 0.3664981 ],
       [ 1.10834955],
       [-0.92733456]]), 'W2': np.array([[ 0.79014539, -1.55941291, -0.65      , -0.68068302,  1.39036133,
        -0.14384368]]), 'b2': np.array([[0.80513831]])}

#Eppaisseur tuyau, trou tuyau = 3, espace tuyau = 25, nbr generation = 20

best_param = {'W1': np.array([[-0.416015  , -0.01945844, -0.22081255],
       [-0.2       , -0.69832457, -0.72293337],
       [-0.19717337,  0.44042862,  0.02591887],
       [-1.21493601,  0.28428758, -0.2545396 ],
       [-0.08328734, -0.85722845, -0.31933287],
       [ 1.29110006,  0.15879475, -0.6328303 ]]), 'b1': np.array([[ 0.97134392],
       [ 0.50526061],
       [ 0.95888811],
       [ 0.5000742 ],
       [-0.75352754],
       [ 0.15494178]]), 'W2': np.array([[ 0.46995051,  0.76720207,  0.71885452, -1.36853858,  1.40333045,
        -0.71057332]]), 'b2': np.array([[-0.51601382]])}

autre_best_param = {'W1': np.array([[ 1.69706592, -0.93371405, -0.39302167],
       [-1.22645695,  2.22369636, -0.80085034],
       [ 0.1       , -1.02277913,  0.12263302],
       [ 0.56686235, -1.26679861, -0.7       ],
       [ 0.4722713 , -1.56058861,  1.25382737],
       [ 0.21739152,  1.08485295, -0.6255698 ]]), 'b1': np.array([[ 0.47386282],
       [-0.58146216],
       [ 1.50163351],
       [ 1.07011434],
       [-0.17781861],
       [ 0.4       ]]), 'W2': np.array([[-1.29611337, -1.77361835,  1.06231767, -0.50303023,  0.24033666,
        -1.04948196]]), 'b2': np.array([[1.5645592]])}

tab = [Oiseau(2,best_param), Oiseau(3,autre_best_param)]
faire_jouer_IA(tab)

pygame.quit()