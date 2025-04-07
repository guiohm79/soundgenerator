"""
ANNULEUR DE BRUIT PRO - VERSION PYO/ASIO
Un suppresseur de bruit par inversion de phase utilisant Pyo pour une latence minimale
et des performances optimales avec les pilotes ASIO.

PARTIE 1: Importations et Classe principale
"""

import time
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
from matplotlib.figure import Figure
import queue
import os
import sys

# Vérification et installation de Pyo si nécessaire
try:
    import pyo
except ImportError:
    print("La bibliothèque Pyo n'est pas installée. Tentative d'installation...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyo"])
    import pyo


class AnuleurDeBruitPyoASIO:
    """
    Classe principale pour le suppresseur de bruit utilisant Pyo avec ASIO
    """
    def __init__(self, 
                 taux_echantillonnage=96000,
                 taille_tampon=64,
                 device_entree=None,
                 device_sortie=None):
        """
        Initialise le système d'annulation de bruit avec Pyo et ASIO
        
        Args:
            taux_echantillonnage: Fréquence d'échantillonnage en Hz
            taille_tampon: Nombre d'échantillons par trame (buffer size)
            device_entree: Nom ou index du périphérique d'entrée
            device_sortie: Nom ou index du périphérique de sortie
        """
        self.taux_echantillonnage = taux_echantillonnage
        self.taille_tampon = taille_tampon
        self.device_entree = device_entree
        self.device_sortie = device_sortie
        
        # Paramètres pour le filtrage et le traitement
        self.delai_ms = 8.0  # Délai en millisecondes (à ajuster)
        self.gain = 0.95  # Gain à appliquer au signal inversé
        
        # Paramètres du filtre passe-bande
        self.freq_basse = 50  # Hz
        self.freq_haute = 4000  # Hz
        self.freq_centre = (self.freq_basse + self.freq_haute) / 2
        self.q_factor = self.freq_centre / (self.freq_haute - self.freq_basse)
        
        # État d'exécution
        self.en_cours = False
        self.serveur = None
        self.thread_traitement = None
        
        # Données pour affichage graphique (tampon circulaire)
        self.buffer_size = 1000
        self.donnees_entree = np.zeros(self.buffer_size)
        self.donnees_sortie = np.zeros(self.buffer_size)
        self.buffer_position = 0
        
        # Mesure des performances
        self.cpu_usage = 0
        self.latence_mesuree = 0
        
        # Créer la chaîne de traitement audio
        self.input_stream = None
        self.output_stream = None
        self.filtre = None
        self.inverseur = None
        self.delai = None
        self.mixeur = None
    
    @staticmethod
    def liste_peripheriques():
        """
        Retourne la liste des périphériques audio disponibles via Pyo
        """
        try:
            # Créer un serveur temporaire pour lister les périphériques
            temp_server = pyo.Server(duplex=0)
            
            # Obtenir la liste des périphériques
            devices_in = temp_server.getInputDevices()
            devices_out = temp_server.getOutputDevices()
            
            # Formater la liste des périphériques pour l'interface
            device_list = []
            device_info = {}
            
            # Périphériques d'entrée
            for i, dev in enumerate(devices_in[1]):
                name = dev
                device_list.append((i, f"{name} (Entrée)"))
                device_info[i] = {
                    'name': name,
                    'inputs': True,
                    'outputs': False,
                }
            
            # Périphériques de sortie
            for i, dev in enumerate(devices_out[1]):
                name = dev
                device_list.append((i + len(devices_in[1]), f"{name} (Sortie)"))
                device_info[i + len(devices_in[1])] = {
                    'name': name,
                    'inputs': False,
                    'outputs': True,
                }
            
            # Nettoyer le serveur temporaire
            temp_server.shutdown()
            
            return device_list, device_info
            
        except Exception as e:
            print(f"Erreur lors de la liste des périphériques: {e}")
            return [], {}
    
    def demarrer(self):
        """
        Démarre le traitement audio avec Pyo/ASIO
        """
        if not self.en_cours:
            try:
                print("Démarrage du serveur audio Pyo avec ASIO...")
                
                # Initialiser le serveur Pyo avec ASIO si disponible
                self.serveur = pyo.Server(
                    sr=self.taux_echantillonnage,
                    buffersize=self.taille_tampon,
                    duplex=1,  # Activer entrée et sortie
                    audio='asio' if 'asio' in pyo.pa_get_host_apis() else 'portaudio',
                    ichnls=1,  # 1 canal d'entrée (mono)
                    ochnls=2,  # 2 canaux de sortie (stéréo)
                    winhost="wasapi" if 'asio' not in pyo.pa_get_host_apis() else None
                )
                
                # Configurer les périphériques si spécifiés
                if self.device_entree is not None and self.device_sortie is not None:
                    print(f"Configuration des périphériques: IN={self.device_entree}, OUT={self.device_sortie}")
                    self.serveur.setInputDevice(self.device_entree)
                    self.serveur.setOutputDevice(self.device_sortie)
                
                # Démarrer le serveur
                self.serveur.boot()
                
                # Démarrer le stream audio
                self.serveur.start()
                
                # Configuration de la chaîne de traitement audio
                self._configurer_traitement()
                
                # Activer la récupération des données pour l'affichage
                self._configurer_affichage()
                
                self.en_cours = True
                
                # Récupérer les informations de latence
                latence_entree = self.serveur.getInputLatency() * 1000  # en ms
                latence_sortie = self.serveur.getOutputLatency() * 1000  # en ms
                self.latence_mesuree = latence_entree + latence_sortie
                
                print(f"Serveur audio démarré avec succès!")
                print(f"Taux d'échantillonnage: {self.serveur.getSamplingRate()} Hz")
                print(f"Taille du tampon: {self.serveur.getBufferSize()} échantillons")
                print(f"Latence d'entrée: {latence_entree:.2f} ms")
                print(f"Latence de sortie: {latence_sortie:.2f} ms")
                print(f"Latence totale: {self.latence_mesuree:.2f} ms")
                
                # Démarrer un thread pour surveiller l'utilisation CPU
                self.thread_traitement = threading.Thread(target=self._surveillance_cpu, daemon=True)
                self.thread_traitement.start()
                
                return True
                
            except Exception as e:
                print(f"Erreur lors du démarrage du serveur audio: {str(e)}")
                if self.serveur:
                    self.serveur.shutdown()
                    self.serveur = None
                self.en_cours = False
                raise e
        
        return False
    def _configurer_traitement(self):
        """
        Configure la chaîne de traitement audio avec Pyo
        """
        if not self.serveur:
            return
        
        # Créer le flux d'entrée
        self.input_stream = pyo.Input(chnl=0)
        
        # Appliquer un filtre passe-bande pour isoler les fréquences d'intérêt
        self.filtre = pyo.Biquad(
            self.input_stream, 
            freq=self.freq_centre, 
            q=self.q_factor, 
            type=2  # Type 2 = passe-bande
        )
        
        # Inverser le signal (multiplier par -1)
        self.inverseur = self.filtre * -self.gain
        
        # Appliquer un délai pour compenser la latence
        self.delai = pyo.Delay(
            self.inverseur,
            delay=self.delai_ms / 1000.0,  # Convertir ms en secondes
            feedback=0
        )
        
        # Mélanger le signal original avec le signal inversé et retardé
        self.mixeur = self.input_stream + self.delai
        
        # Envoyer vers la sortie
        self.mixeur.out()
    
    def _configurer_affichage(self):
        """
        Configure la récupération des données pour l'affichage
        """
        if not self.serveur:
            return
        
        # Créer des objets pour récupérer les données d'entrée et de sortie
        self.tableau_entree = pyo.NewTable(self.buffer_size / self.taux_echantillonnage)
        self.tableau_sortie = pyo.NewTable(self.buffer_size / self.taux_echantillonnage)
        
        # Enregistrer l'entrée et la sortie dans les tableaux
        self.enregistreur_entree = pyo.TableRec(self.input_stream, self.tableau_entree, 
                                              fadeout=0, loop=True)
        self.enregistreur_sortie = pyo.TableRec(self.mixeur, self.tableau_sortie, 
                                             fadeout=0, loop=True)
        
        # Démarrer l'enregistrement
        self.enregistreur_entree.play()
        self.enregistreur_sortie.play()
        
        # Fonction de rappel pour récupérer les données
        def callback_donnees():
            if self.en_cours:
                # Récupérer les données des tableaux
                self.donnees_entree = np.array(self.tableau_entree.getTable())
                self.donnees_sortie = np.array(self.tableau_sortie.getTable())
                
                # Programmer le prochain appel (environ 30 fps)
                self.serveur.CallAfter(0.033, callback_donnees)
        
        # Démarrer la récupération des données
        self.serveur.CallAfter(0.1, callback_donnees)
    
    def _surveillance_cpu(self):
        """
        Surveillance continue de l'utilisation CPU
        """
        while self.en_cours:
            if self.serveur:
                self.cpu_usage = self.serveur.getStreamCpuLoad() * 100
            time.sleep(1)
    
    def arreter(self):
        """
        Arrête le traitement audio
        """
        if self.en_cours:
            self.en_cours = False
            
            if self.serveur:
                print("Arrêt du serveur audio...")
                try:
                    # Arrêter toute la chaîne de traitement
                    if hasattr(self, 'mixeur') and self.mixeur:
                        self.mixeur.stop()
                    if hasattr(self, 'delai') and self.delai:
                        self.delai.stop()
                    if hasattr(self, 'inverseur') and self.inverseur:
                        self.inverseur.stop()
                    if hasattr(self, 'filtre') and self.filtre:
                        self.filtre.stop()
                    if hasattr(self, 'input_stream') and self.input_stream:
                        self.input_stream.stop()
                    
                    # Arrêter les enregistreurs
                    if hasattr(self, 'enregistreur_entree') and self.enregistreur_entree:
                        self.enregistreur_entree.stop()
                    if hasattr(self, 'enregistreur_sortie') and self.enregistreur_sortie:
                        self.enregistreur_sortie.stop()
                    
                    # Arrêter et fermer le serveur
                    self.serveur.stop()
                    self.serveur.shutdown()
                    self.serveur = None
                except Exception as e:
                    print(f"Erreur lors de l'arrêt du serveur: {e}")
            
            # Attendre la fin du thread de surveillance
            if self.thread_traitement and self.thread_traitement.is_alive():
                self.thread_traitement.join(timeout=1)
            
            print("Serveur audio arrêté")
    
    def regler_parametres(self, delai=None, gain=None, freq_basse=None, freq_haute=None):
        """
        Ajuste les paramètres du traitement en temps réel
        """
        if not self.en_cours:
            return
        
        if delai is not None:
            self.delai_ms = delai
            if self.delai:
                self.delai.setDelay(self.delai_ms / 1000.0)
        
        if gain is not None:
            self.gain = gain
            # Pour mettre à jour le gain, on doit reconfigurer l'inverseur
            if self.inverseur and self.filtre:
                self.inverseur = self.filtre * -self.gain
                if self.delai:
                    self.delai.setInput(self.inverseur)
        
        if freq_basse is not None or freq_haute is not None:
            if freq_basse is not None:
                self.freq_basse = freq_basse
            if freq_haute is not None:
                self.freq_haute = freq_haute
            
            # Recalculer les paramètres du filtre
            self.freq_centre = (self.freq_basse + self.freq_haute) / 2
            self.q_factor = self.freq_centre / (self.freq_haute - self.freq_basse)
            
            # Mettre à jour le filtre
            if self.filtre:
                self.filtre.setFreq(self.freq_centre)
                self.filtre.setQ(self.q_factor)
    
    def calibrer(self):
        """
        Calibration automatique du délai et du gain
        """
        if not self.en_cours:
            return self.delai_ms, self.gain
        
        print("Calibration en cours... Veuillez faire du bruit constant.")
        
        # Utiliser la latence mesurée du serveur pour estimer le délai
        latence_entree = self.serveur.getInputLatency() * 1000  # en ms
        latence_sortie = self.serveur.getOutputLatency() * 1000  # en ms
        latence_totale = latence_entree + latence_sortie
        
        # Ajouter une marge pour compenser d'autres sources de latence
        self.delai_ms = latence_totale + 2.0  # +2ms de marge
        
        # Mettre à jour le délai
        if self.delai:
            self.delai.setDelay(self.delai_ms / 1000.0)
        
        # Le gain reste à 0.95 pour éviter les oscillations
        self.gain = 0.95
        
        print(f"Calibration terminée: délai={self.delai_ms:.1f}ms, gain={self.gain}")
        return self.delai_ms, self.gain
    
    def fermer(self):
        """
        Ferme proprement toutes les ressources
        """
        self.arreter()

class InterfaceAnnuleurPyo(tk.Tk):
    """
    Interface graphique pour l'annuleur de bruit avec Pyo
    """
    def __init__(self):
        super().__init__()
        
        self.title("Annuleur de Bruit Pro - Version ASIO Turbo")
        self.geometry("950x750")
        
        # Thème moderne
        self.style = ttk.Style()
        if 'clam' in self.style.theme_names():
            self.style.theme_use('clam')
        
        # Configurer des couleurs modernes
        self.couleur_bg = "#2E3440"  # Fond sombre
        self.couleur_fg = "#D8DEE9"  # Texte clair
        self.couleur_accent = "#88C0D0"  # Couleur d'accent
        self.couleur_warning = "#EBCB8B"  # Jaune avertissement
        self.couleur_success = "#A3BE8C"  # Vert succès
        
        # Appliquer la palette de couleurs
        self.configure(bg=self.couleur_bg)
        
        # Styles personnalisés
        self.style.configure("TFrame", background=self.couleur_bg)
        self.style.configure("TLabelframe", background=self.couleur_bg, foreground=self.couleur_fg)
        self.style.configure("TLabelframe.Label", background=self.couleur_bg, foreground=self.couleur_accent, font=('Arial', 11, 'bold'))
        self.style.configure("TLabel", background=self.couleur_bg, foreground=self.couleur_fg, font=('Arial', 10))
        self.style.configure("TButton", font=('Arial', 10, 'bold'))
        self.style.map("TButton", background=[("active", self.couleur_accent)])
        self.style.configure("Success.TButton", background=self.couleur_success)
        self.style.configure("Warning.TButton", background=self.couleur_warning)
        
        # Créer l'instance d'annuleur
        self.annuleur = AnuleurDeBruitPyoASIO(taux_echantillonnage=96000, taille_tampon=64)
        
        # Créer l'interface
        self._create_widgets()
        
        # Animation pour la visualisation
        self.ani = None
        self._start_animation()
        
        # Mise à jour des performances
        self.after(1000, self._update_performances)
        
        # S'assurer que tout est fermé proprement
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _create_widgets(self):
        """
        Crée tous les widgets de l'interface
        """
        # Frame principale
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ===== En-tête avec logo =====
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, padx=5, pady=5)
        
        title_label = ttk.Label(header_frame, text="ANNULEUR DE BRUIT PRO - VERSION TURBO", 
                               font=('Arial', 16, 'bold'), foreground=self.couleur_accent)
        title_label.pack(side=tk.LEFT, padx=5)
        
        version_label = ttk.Label(header_frame, text="v3.0", font=('Arial', 8))
        version_label.pack(side=tk.LEFT)
        
        # Indicateurs de performance
        self.perf_frame = ttk.Frame(header_frame)
        self.perf_frame.pack(side=tk.RIGHT, padx=5)
        
        self.latence_label = ttk.Label(self.perf_frame, text="Latence: -- ms", font=('Arial', 9))
        self.latence_label.pack(side=tk.RIGHT, padx=10)
        
        self.cpu_label = ttk.Label(self.perf_frame, text="CPU: --%", font=('Arial', 9))
        self.cpu_label.pack(side=tk.RIGHT, padx=10)
        
        # ===== Section des périphériques =====
        devices_frame = ttk.LabelFrame(main_frame, text="Périphériques Audio ASIO")
        devices_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Récupérer la liste des périphériques
        self.devices, self.device_info = self.annuleur.liste_peripheriques()
        
        # Configuration en deux colonnes
        devices_inner = ttk.Frame(devices_frame)
        devices_inner.pack(fill=tk.X, padx=5, pady=5)
        devices_inner.columnconfigure(0, weight=1)
        devices_inner.columnconfigure(1, weight=1)
        
        # Entrée audio
        entree_frame = ttk.Frame(devices_inner)
        entree_frame.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W+tk.E)
        
        ttk.Label(entree_frame, text="Entrée audio:").pack(anchor=tk.W, padx=5, pady=2)
        self.entree_var = tk.StringVar()
        self.combo_entree = ttk.Combobox(entree_frame, textvariable=self.entree_var, state="readonly", width=35)
        self.combo_entree['values'] = [d[1] for d in self.devices if "Entrée" in d[1]]
        self.combo_entree.pack(fill=tk.X, padx=5, pady=2)
        if self.combo_entree['values']:
            self.combo_entree.current(0)
        
        # Sortie audio
        sortie_frame = ttk.Frame(devices_inner)
        sortie_frame.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        ttk.Label(sortie_frame, text="Sortie audio:").pack(anchor=tk.W, padx=5, pady=2)
        self.sortie_var = tk.StringVar()
        self.combo_sortie = ttk.Combobox(sortie_frame, textvariable=self.sortie_var, state="readonly", width=35)
        self.combo_sortie['values'] = [d[1] for d in self.devices if "Sortie" in d[1]]
        self.combo_sortie.pack(fill=tk.X, padx=5, pady=2)
        if self.combo_sortie['values']:
            self.combo_sortie.current(0)
        
        # ===== Section des contrôles =====
        controls_frame = ttk.LabelFrame(main_frame, text="Contrôles")
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Boutons
        buttons_frame = ttk.Frame(controls_frame)
        buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_button = ttk.Button(buttons_frame, text="▶ Démarrer", 
                                    command=self._start_processing, style="Success.TButton")
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.stop_button = ttk.Button(buttons_frame, text="⏹ Arrêter", 
                                   command=self._stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.calib_button = ttk.Button(buttons_frame, text="🔄 Calibrer", 
                                     command=self._calibrate, style="TButton")
        self.calib_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # ===== Section des paramètres =====
        params_frame = ttk.LabelFrame(main_frame, text="Paramètres")
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Séparer en deux colonnes
        params_left_frame = ttk.Frame(params_frame)
        params_left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        params_right_frame = ttk.Frame(params_frame)
        params_right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Délai - colonne gauche
        delai_frame = ttk.Frame(params_left_frame)
        delai_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(delai_frame, text="Délai de compensation (ms):").pack(anchor=tk.W)
        
        delai_controls = ttk.Frame(delai_frame)
        delai_controls.pack(fill=tk.X)
        
        self.delai_var = tk.DoubleVar(value=8.0)
        self.scale_delai = ttk.Scale(
            delai_controls, from_=0, to=50, variable=self.delai_var, 
            command=lambda x: self._update_param_label('delai')
        )
        self.scale_delai.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.delai_label = ttk.Label(delai_controls, text="8.0 ms", width=8)
        self.delai_label.pack(side=tk.RIGHT, padx=5)
        
        # Gain - colonne gauche
        gain_frame = ttk.Frame(params_left_frame)
        gain_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(gain_frame, text="Gain:").pack(anchor=tk.W)
        
        gain_controls = ttk.Frame(gain_frame)
        gain_controls.pack(fill=tk.X)
        
        self.gain_var = tk.DoubleVar(value=0.95)
        self.scale_gain = ttk.Scale(
            gain_controls, from_=0, to=2, variable=self.gain_var, 
            command=lambda x: self._update_param_label('gain')
        )
        self.scale_gain.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.gain_label = ttk.Label(gain_controls, text="0.95", width=8)
        self.gain_label.pack(side=tk.RIGHT, padx=5)
        
        # Fréquence basse - colonne droite
        freq_basse_frame = ttk.Frame(params_right_frame)
        freq_basse_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(freq_basse_frame, text="Fréquence basse (Hz):").pack(anchor=tk.W)
        
        freq_basse_controls = ttk.Frame(freq_basse_frame)
        freq_basse_controls.pack(fill=tk.X)
        
        self.freq_basse_var = tk.DoubleVar(value=50)
        self.scale_freq_basse = ttk.Scale(
            freq_basse_controls, from_=20, to=1000, variable=self.freq_basse_var, 
            command=lambda x: self._update_param_label('freq_basse')
        )
        self.scale_freq_basse.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.freq_basse_label = ttk.Label(freq_basse_controls, text="50 Hz", width=8)
        self.freq_basse_label.pack(side=tk.RIGHT, padx=5)
        
        # Fréquence haute - colonne droite
        freq_haute_frame = ttk.Frame(params_right_frame)
        freq_haute_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(freq_haute_frame, text="Fréquence haute (Hz):").pack(anchor=tk.W)
        
        freq_haute_controls = ttk.Frame(freq_haute_frame)
        freq_haute_controls.pack(fill=tk.X)
        
        self.freq_haute_var = tk.DoubleVar(value=4000)
        self.scale_freq_haute = ttk.Scale(
            freq_haute_controls, from_=1000, to=20000, variable=self.freq_haute_var, 
            command=lambda x: self._update_param_label('freq_haute')
        )
        self.scale_freq_haute.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.freq_haute_label = ttk.Label(freq_haute_controls, text="4000 Hz", width=8)
        self.freq_haute_label.pack(side=tk.RIGHT, padx=5)
        
        # Bouton Appliquer centré en bas
        apply_frame = ttk.Frame(params_frame)
        apply_frame.pack(fill=tk.X, pady=10)
        
        self.apply_button = ttk.Button(
            apply_frame, 
            text="✓ Appliquer", 
            command=self._apply_params,
            style="Success.TButton"
        )
        self.apply_button.pack(pady=5)
        
        # ===== Section de visualisation =====
        viz_frame = ttk.LabelFrame(main_frame, text="Visualisation en temps réel")
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Utiliser Figure pour une meilleure personnalisation
        self.fig = Figure(figsize=(8, 4), dpi=100, facecolor=self.couleur_bg)
        
        # Configurer les axes avec des couleurs adaptées au thème sombre
        self.ax1 = self.fig.add_subplot(211)
        self.ax1.set_title("Signal d'entrée", color=self.couleur_fg)
        self.ax1.set_ylim(-1, 1)
        self.ax1.set_xlim(0, 1000)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_facecolor("#3B4252")  # Fond un peu plus clair
        self.ax1.tick_params(axis='x', colors=self.couleur_fg)
        self.ax1.tick_params(axis='y', colors=self.couleur_fg)
        
        self.ax2 = self.fig.add_subplot(212)
        self.ax2.set_title("Signal inversé + compensation", color=self.couleur_fg)
        self.ax2.set_ylim(-1, 1)
        self.ax2.set_xlim(0, 1000)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_facecolor("#3B4252")
        self.ax2.tick_params(axis='x', colors=self.couleur_fg)
        self.ax2.tick_params(axis='y', colors=self.couleur_fg)
        
        # Line plots avec de jolies couleurs
        x = np.arange(1000)
        self.line1, = self.ax1.plot(x, np.zeros(1000), color='#BF616A', lw=1.5)  # Rouge nordique
        self.line2, = self.ax2.plot(x, np.zeros(1000), color='#A3BE8C', lw=1.5)  # Vert nordique
        
        self.fig.tight_layout(pad=3.0)
        
        # Intégrer la figure dans tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Barre d'état
        self.status_frame = ttk.Frame(main_frame)
        self.status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_label = ttk.Label(
            self.status_frame, 
            text="Prêt. Sélectionnez vos périphériques et cliquez sur Démarrer.",
            font=('Arial', 9, 'italic')
        )
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Label avec lien vers l'aide
        self.help_label = ttk.Label(
            self.status_frame, 
            text="Besoin d'aide ?",
            foreground=self.couleur_accent,
            cursor="hand2",
            font=('Arial', 9, 'underline')
        )
        self.help_label.pack(side=tk.RIGHT, padx=5)
        self.help_label.bind("<Button-1>", self._show_help)
    
    def _update_param_label(self, param_name):
        """Met à jour les labels des sliders"""
        if param_name == 'delai':
            value = self.delai_var.get()
            self.delai_label.config(text=f"{value:.1f} ms")
        elif param_name == 'gain':
            value = self.gain_var.get()
            self.gain_label.config(text=f"{value:.2f}")
        elif param_name == 'freq_basse':
            value = self.freq_basse_var.get()
            self.freq_basse_label.config(text=f"{int(value)} Hz")
        elif param_name == 'freq_haute':
            value = self.freq_haute_var.get()
            self.freq_haute_label.config(text=f"{int(value)} Hz")
    
    def _start_processing(self):
        """Démarre le traitement audio"""
        try:
            # Récupérer les indices de périphériques sélectionnés
            entree_idx = None
            sortie_idx = None
            
            for idx, name in self.devices:
                if name == self.entree_var.get():
                    entree_idx = idx
                if name == self.sortie_var.get():
                    sortie_idx = idx
            
            if entree_idx is None or sortie_idx is None:
                messagebox.showerror("Erreur", "Veuillez sélectionner des périphériques d'entrée et de sortie.")
                return
            
            # Configurer les périphériques
            self.annuleur.device_entree = entree_idx
            self.annuleur.device_sortie = sortie_idx
            
            # Appliquer les paramètres actuels
            self._apply_params()
            
            # Démarrer le traitement
            if self.annuleur.demarrer():
                # Mettre à jour l'interface
                self.start_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.NORMAL)
                self.combo_entree.config(state=tk.DISABLED)
                self.combo_sortie.config(state=tk.DISABLED)
                
                # Mettre à jour le status
                self.status_label.config(
                    text=f"Traitement en cours. Latence estimée: {self.annuleur.latence_mesuree:.1f} ms"
                )
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de démarrer: {str(e)}")
    
    def _stop_processing(self):
        """Arrête le traitement audio"""
        self.annuleur.arreter()
        
        # Mettre à jour l'interface
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.combo_entree.config(state="readonly")
        self.combo_sortie.config(state="readonly")
        
        # Mettre à jour le status
        self.status_label.config(text="Traitement arrêté.")
    
    def _apply_params(self):
        """Applique les paramètres actuels"""
        self.annuleur.regler_parametres(
            delai=self.delai_var.get(),
            gain=self.gain_var.get(),
            freq_basse=self.freq_basse_var.get(),
            freq_haute=self.freq_haute_var.get()
        )
        
        self.status_label.config(text="Paramètres appliqués.")
    
    def _calibrate(self):
        """Lance la calibration automatique"""
        if not self.annuleur.en_cours:
            messagebox.showinfo("Calibration", "Veuillez d'abord démarrer le traitement avant de calibrer.")
            return
            
        # Lancer la calibration
        try:
            delai, gain = self.annuleur.calibrer()
            
            # Mettre à jour les sliders
            self.delai_var.set(delai)
            self.gain_var.set(gain)
            
            # Mettre à jour les labels
            self._update_param_label('delai')
            self._update_param_label('gain')
            
            # Message de réussite
            self.status_label.config(text="Calibration terminée avec succès !")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Échec de la calibration: {str(e)}")
    
    def _update_performances(self):
        """Met à jour les indicateurs de performances"""
        if self.annuleur.en_cours:
            # Mettre à jour les labels
            self.latence_label.config(text=f"Latence: {self.annuleur.latence_mesuree:.1f} ms")
            self.cpu_label.config(text=f"CPU: {self.annuleur.cpu_usage:.1f}%")
            
            # Colorer selon les performances
            if self.annuleur.latence_mesuree < 10:
                self.latence_label.config(foreground=self.couleur_success)
            elif self.annuleur.latence_mesuree < 20:
                self.latence_label.config(foreground=self.couleur_accent)
            else:
                self.latence_label.config(foreground=self.couleur_warning)
                
            if self.annuleur.cpu_usage < 30:
                self.cpu_label.config(foreground=self.couleur_success)
            elif self.annuleur.cpu_usage < 70:
                self.cpu_label.config(foreground=self.couleur_accent)
            else:
                self.cpu_label.config(foreground=self.couleur_warning)
        else:
            # Réinitialiser les labels
            self.latence_label.config(text="Latence: -- ms", foreground=self.couleur_fg)
            self.cpu_label.config(text="CPU: --%", foreground=self.couleur_fg)
        
        # Programmer la prochaine mise à jour
        self.after(1000, self._update_performances)
    
    def _update_plot(self, frame):
        """Met à jour le graphique en temps réel"""
        if self.annuleur.en_cours:
            self.line1.set_ydata(self.annuleur.donnees_entree)
            self.line2.set_ydata(self.annuleur.donnees_sortie)
        return self.line1, self.line2
    
    def _start_animation(self):
        """Démarre l'animation pour le graphique en temps réel"""
        self.ani = animation.FuncAnimation(
            self.fig, self._update_plot, interval=33,  # ~30 fps
            blit=True, cache_frame_data=False
        )
    
    def _show_help(self, event=None):
        """Affiche l'aide sur l'utilisation de l'application"""
        help_text = """
        Guide d'utilisation de l'Annuleur de Bruit Pro - Version ASIO Turbo
        
        1. Sélection des périphériques:
           - Choisissez votre micro comme entrée
           - Choisissez vos haut-parleurs comme sortie
        
        2. Contrôles:
           - Démarrer: lance le traitement audio
           - Arrêter: interrompt le traitement
           - Calibrer: ajuste automatiquement les paramètres
        
        3. Paramètres:
           - Délai: compense la latence du système (3-10ms recommandé)
           - Gain: intensité de l'annulation (0.8-1.0 recommandé)
           - Fréquences: ajuste la plage de fréquences à traiter
        
        4. Astuces pour une latence minimale:
           - Utilisez de vrais périphériques ASIO si disponibles
           - Sélectionnez un taux d'échantillonnage élevé (96kHz)
           - Fermez les applications gourmandes en CPU
           - Pour les bruits constants (ventilateurs, ronronnements), utilisez un délai plus élevé
           - Pour les bruits variables, privilégiez un délai plus court
        
        Bon travail de suppression de bruit !
        """
        
        # Créer une fenêtre d'aide avec un style moderne
        help_window = tk.Toplevel(self)
        help_window.title("Aide - Annuleur de Bruit Pro")
        help_window.geometry("500x450")
        help_window.configure(bg=self.couleur_bg)
        
        # Titre
        title_label = ttk.Label(
            help_window, 
            text="Guide d'utilisation", 
            font=('Arial', 14, 'bold'),
            foreground=self.couleur_accent,
            background=self.couleur_bg
        )
        title_label.pack(pady=10)
        
        # Zone de texte avec scrollbar
        text_frame = ttk.Frame(help_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_box = tk.Text(
            text_frame, 
            wrap=tk.WORD, 
            bg="#3B4252", 
            fg=self.couleur_fg,
            font=('Arial', 10),
            bd=0,
            padx=10,
            pady=10,
            highlightthickness=0
        )
        text_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configurer la scrollbar
        scrollbar.config(command=text_box.yview)
        text_box.config(yscrollcommand=scrollbar.set)
        
        # Insérer le texte d'aide
        text_box.insert(tk.END, help_text)
        text_box.config(state=tk.DISABLED)  # Rendre en lecture seule
        
        # Bouton de fermeture
        close_button = ttk.Button(
            help_window, 
            text="Fermer", 
            command=help_window.destroy
        )
        close_button.pack(pady=10)
    
    def _on_closing(self):
        """Ferme proprement l'application"""
        # Arrêter l'animation
        if self.ani:
            self.ani.event_source.stop()
        
        # Arrêter le traitement
        if hasattr(self, 'annuleur'):
            self.annuleur.fermer()
            
        # Fermer la fenêtre
        self.destroy()


if __name__ == "__main__":
    print("=== Annuleur de Bruit Pro - Version ASIO Turbo ===")
    print("Initialisation du système d'annulation de bruit avec Pyo...")
    
    # Vérifier si Pyo est installé
    try:
        import pyo
        print(f"Pyo version {pyo.getVersion()} trouvée.")
        
        # Lister les APIs audio disponibles
        apis = pyo.pa_get_host_apis()
        print("\n===== APIs AUDIO DISPONIBLES =====")
        for i, api in enumerate(apis):
            print(f"[{i}] {api}")
        
        # Vérifier si ASIO est disponible
        if 'asio' in apis:
            print("\nAPI ASIO détectée ! Performances optimales disponibles. 🎉")
        else:
            print("\nAPI ASIO non détectée. Les performances seront limitées.")
            print("Vous pouvez installer ASIO4ALL pour de meilleures performances.")
        
    except ImportError:
        print("La bibliothèque Pyo n'est pas installée.")
        print("Installation automatique en cours...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyo"])
        print("Pyo installé avec succès ! Redémarrez l'application.")
        sys.exit(0)
    
    # Démarrer l'interface
    app = InterfaceAnnuleurPyo()
    app.mainloop()