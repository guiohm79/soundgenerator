import numpy as np
import sounddevice as sd
import time
from scipy import signal
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import queue
from matplotlib.figure import Figure
import concurrent.futures
import colorsys

class AnuleurDeBruitASIO:
    def __init__(self, 
                 taux_echantillonnage=96000,
                 taille_tampon=64,  # Taille réduite pour moins de latence
                 device_entree=None,
                 device_sortie=None,
                 api='asio'):
        """
        Initialise le système d'annulation de bruit avec support ASIO
        
        Args:
            taux_echantillonnage: Fréquence d'échantillonnage en Hz
            taille_tampon: Nombre d'échantillons par trame (réduit pour moins de latence)
            device_entree: Nom ou index du périphérique d'entrée
            device_sortie: Nom ou index du périphérique de sortie
            api: API audio à utiliser ('asio' par défaut)
        """
        self.taux_echantillonnage = taux_echantillonnage
        self.taille_tampon = taille_tampon
        self.device_entree = device_entree
        self.device_sortie = device_sortie
        self.api = api
        
        # Paramètres pour le filtrage et le traitement
        self.delai_compensation = 0  # Délai pour compenser la latence du système
        self.gain = 1.0  # Gain à appliquer au signal inversé
        
        # Filtres pour améliorer la performance
        self.ordre_filtre = 2  # Ordre réduit pour moins de latence
        self.freq_coupure_basse = 50  # Hz
        self.freq_coupure_haute = 4000  # Hz
        self.b, self.a = self._creer_filtre()
        
        # État du filtre
        self.z = signal.lfilter_zi(self.b, self.a) * 0
        
        # Tampon pour stocker les échantillons précédents (pour le délai)
        self.tampon_historique = np.zeros(self.taux_echantillonnage // 2)  # 0.5 seconde d'historique (réduit)
        self.position_tampon = 0
        
        # Données pour affichage graphique
        self.donnees_entree = np.zeros(1000)  # Plus de points pour un affichage plus fluide
        self.donnees_sortie = np.zeros(1000)
        
        # Mesures de performance
        self.temps_traitement = []
        self.cpu_usage = 0
        
        # File d'attente pour les données audio
        self.queue_audio = queue.Queue(maxsize=10)  # Taille réduite pour moins de latence
        
        # Traitement parallèle
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
        # État d'exécution
        self.en_cours = False
        self.thread_traitement = None
        self.stream = None
        
        # Mode d'optimisation auto
        self.mode_faible_latence = True
    
    def _creer_filtre(self):
        """
        Crée le filtre passe-bande avec mémorisation de l'état pour réduire la latence
        """
        return signal.butter(
            self.ordre_filtre, 
            [self.freq_coupure_basse/(self.taux_echantillonnage/2), 
             self.freq_coupure_haute/(self.taux_echantillonnage/2)], 
            btype='band'
        )
    
    def callback_audio(self, indata, outdata, frames, time_info, status):
        """
        Callback appelé par sounddevice pour traiter l'audio
        """
        # Note : 'time' dans le callback est en réalité un objet time_info, pas le module time
        # Utilisons le module time importé globalement
        debut_traitement = time.time()
        
        if status:
            print(f"Statut audio: {status}")
        
        # Extraire les données audio mono (prendre la première colonne si stéréo)
        donnees_audio = indata[:, 0] if indata.ndim > 1 else indata[:]
        
        # Stocker pour visualisation avec échantillonnage réduit pour économiser le CPU
        if np.random.random() < 0.8:  # 80% du temps, on ignore l'échantillon pour l'affichage
            self.donnees_entree = np.roll(self.donnees_entree, -len(donnees_audio))
            if len(donnees_audio) < len(self.donnees_entree):
                self.donnees_entree[-len(donnees_audio):] = donnees_audio
            else:
                self.donnees_entree = donnees_audio[-len(self.donnees_entree):]
        
        # Stocker les données dans le tampon d'historique circulaire de manière optimisée
        taille_tampon = len(self.tampon_historique)
        indices = np.mod(np.arange(self.position_tampon, self.position_tampon + len(donnees_audio)), taille_tampon)
        self.tampon_historique[indices] = donnees_audio
        self.position_tampon = (self.position_tampon + len(donnees_audio)) % taille_tampon
        
        # Appliquer un filtre passe-bande avec état mémorisé pour réduire la latence
        donnees_filtrees, self.z = signal.lfilter(self.b, self.a, donnees_audio, zi=self.z)
        
        # Inverser la phase (multiplier par -1)
        donnees_inversees = -self.gain * donnees_filtrees
        
        # Compensation de délai si nécessaire avec calcul vectorisé
        if self.delai_compensation > 0:
            position_delai = (self.position_tampon - self.delai_compensation) % taille_tampon
            indices_retard = np.mod(np.arange(position_delai, position_delai + len(donnees_inversees)), taille_tampon)
            donnees_retardees = self.tampon_historique[indices_retard]
            
            # Mélanger le signal retardé avec le signal inversé
            donnees_sortie = donnees_retardees + donnees_inversees
        else:
            donnees_sortie = donnees_inversees
        
        # Limiter l'amplitude pour éviter la distorsion
        donnees_sortie = np.clip(donnees_sortie, -0.95, 0.95)
        
        # Stocker pour visualisation (même technique d'échantillonnage réduit)
        if np.random.random() < 0.8:
            self.donnees_sortie = np.roll(self.donnees_sortie, -len(donnees_sortie))
            if len(donnees_sortie) < len(self.donnees_sortie):
                self.donnees_sortie[-len(donnees_sortie):] = donnees_sortie
            else:
                self.donnees_sortie = donnees_sortie[-len(self.donnees_sortie):]
        
        # Envoyer les données à la sortie audio (stéréo si nécessaire)
        if outdata.ndim > 1:  # Si c'est multi-canal
            # Remplir chaque canal avec le même signal
            for i in range(outdata.shape[1]):
                outdata[:, i] = donnees_sortie
        else:
            # Cas mono-canal
            outdata.flat[:] = donnees_sortie
            
        # Mesurer le temps de traitement
        fin_traitement = time.time()
        temps_total = (fin_traitement - debut_traitement) * 1000  # en ms
        self.temps_traitement.append(temps_total)
        
        # Garder seulement les 100 dernières mesures
        if len(self.temps_traitement) > 100:
            self.temps_traitement.pop(0)
            
        # Calculer l'utilisation CPU approximative
        temps_disponible = (frames / self.taux_echantillonnage) * 1000  # ms disponibles par callback
        self.cpu_usage = (np.mean(self.temps_traitement) / temps_disponible) * 100
            
    def demarrer(self):
        """
        Démarre le traitement audio avec ASIO
        """
        if not self.en_cours:
            try:
                self.en_cours = True
                
                # Configurer et démarrer le stream
                self.stream = sd.Stream(
                    samplerate=self.taux_echantillonnage,
                    blocksize=self.taille_tampon,
                    device=(self.device_entree, self.device_sortie),
                    channels=(1, 1),  # Mono pour simplifier
                    callback=self.callback_audio,
                    dtype='float32'
                )
                
                # Démarrer le stream
                self.stream.start()
                print(f"Stream audio démarré avec les périphériques: {self.device_entree} -> {self.device_sortie}")
                print(f"Latence d'entrée: {self.stream.latency[0]*1000:.2f}ms, Latence de sortie: {self.stream.latency[1]*1000:.2f}ms")
                print(f"Latence totale estimée: {sum(self.stream.latency)*1000:.2f}ms")
                
                # Ajuster automatiquement le délai de compensation basé sur la latence mesurée
                latence_totale = sum(self.stream.latency)
                self.delai_compensation = int(latence_totale * self.taux_echantillonnage)
                print(f"Délai de compensation auto-configuré: {self.delai_compensation} échantillons")
                
                return True
            except Exception as e:
                self.en_cours = False
                print(f"Erreur lors du démarrage du stream audio: {str(e)}")
                raise e
        
        return False
            
    def arreter(self):
        """
        Arrête le traitement audio
        """
        if self.en_cours:
            self.en_cours = False
            
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
                
            print("Stream audio arrêté")
            
    def calibrer(self):
        """
        Fonction pour calibrer automatiquement le délai et le gain
        """
        print("Calibration en cours... Veuillez faire du bruit constant.")
        
        # Mesurer la latence réelle du système par analyse de corrélation
        # Simuler un signal de test
        duree_test = 0.5  # secondes
        signal_test = np.sin(2 * np.pi * 440 * np.arange(int(duree_test * self.taux_echantillonnage)) / self.taux_echantillonnage)
        
        # Utilisez la latence reportée par le stream pour estimer le délai
        if self.stream and hasattr(self.stream, 'latency'):
            latence_totale = sum(self.stream.latency)
            
            # Ajouter un peu plus pour compenser d'autres latences
            latence_totale += 0.002  # +2ms pour la sécurité
            
            self.delai_compensation = int(latence_totale * self.taux_echantillonnage)
        else:
            # Valeur par défaut si la latence n'est pas disponible
            self.delai_compensation = int(0.010 * self.taux_echantillonnage)  # 10ms de délai
        
        # Optimiser le gain en analysant brièvement le signal d'entrée et de sortie
        if len(self.donnees_entree) > 0 and len(self.donnees_sortie) > 0:
            # Calculer la puissance moyenne du signal d'entrée et de sortie
            puissance_entree = np.mean(np.abs(self.donnees_entree)**2)
            if puissance_entree > 0.001:  # S'assurer qu'il y a du signal
                # Calculer le ratio optimal
                ratio_optimal = 0.95  # Légèrement en dessous de 1 pour éviter les oscillations
                self.gain = ratio_optimal
            else:
                self.gain = 0.95  # Valeur par défaut
        else:
            self.gain = 0.95
            
        print(f"Calibration terminée: délai={self.delai_compensation} échantillons ({self.delai_compensation/self.taux_echantillonnage*1000:.1f}ms), gain={self.gain}")
        return self.delai_compensation, self.gain
        
    def regler_parametres(self, delai=None, gain=None, freq_basse=None, freq_haute=None, 
                          ordre=None, taux_echant=None, taille_tampon=None, mode_faible_latence=None):
        """
        Permet de régler manuellement les paramètres
        """
        recalculer_filtre = False
        redemarrer_stream = False
        
        if delai is not None:
            self.delai_compensation = delai
        if gain is not None:
            self.gain = gain
        if ordre is not None and ordre != self.ordre_filtre:
            self.ordre_filtre = ordre
            recalculer_filtre = True
        if taux_echant is not None and taux_echant != self.taux_echantillonnage:
            self.taux_echantillonnage = taux_echant
            recalculer_filtre = True
            redemarrer_stream = True
        if taille_tampon is not None and taille_tampon != self.taille_tampon:
            self.taille_tampon = taille_tampon
            redemarrer_stream = True
        if mode_faible_latence is not None:
            self.mode_faible_latence = mode_faible_latence
            # Si on active le mode faible latence, réduire l'ordre du filtre et la taille du tampon
            if self.mode_faible_latence and self.ordre_filtre > 2:
                self.ordre_filtre = 2
                recalculer_filtre = True
                if self.taille_tampon > 256:
                    self.taille_tampon = 256
                    redemarrer_stream = True
        
        # Mettre à jour le filtre si nécessaire
        if freq_basse is not None or freq_haute is not None or recalculer_filtre:
            if freq_basse is not None:
                self.freq_coupure_basse = freq_basse
            if freq_haute is not None:
                self.freq_coupure_haute = freq_haute
                
            self.b, self.a = self._creer_filtre()
            self.z = signal.lfilter_zi(self.b, self.a) * 0
            
        # Redémarrer le stream si nécessaire
        if redemarrer_stream and self.en_cours:
            self.arreter()
            time.sleep(0.5)  # Attendre un peu pour s'assurer que tout est fermé
            self.demarrer()
    
    def fermer(self):
        """
        Ferme proprement les flux audio
        """
        self.arreter()
        if hasattr(self, 'executor'):
            self.executor.shutdown()
        
    @staticmethod
    def liste_peripheriques():
        """
        Retourne une liste des périphériques audio disponibles
        """
        try:
            devices = sd.query_devices()
            api_info = sd.query_hostapis()
            
            # Formater la liste des périphériques pour l'interface
            device_list = []
            device_info = {}
            
            for i, dev in enumerate(devices):
                name = dev['name']
                host_api = api_info[dev['hostapi']]['name']
                max_input_channels = dev['max_input_channels']
                max_output_channels = dev['max_output_channels']
                
                # Ajouter les infos
                device_info[i] = {
                    'name': name,
                    'host_api': host_api,
                    'inputs': max_input_channels,
                    'outputs': max_output_channels,
                    'latency': dev.get('default_low_input_latency', 0) + dev.get('default_low_output_latency', 0)
                }
                
                # Ajouter à la liste de sélection avec info de latence
                if max_input_channels > 0:
                    device_list.append((i, f"{name} - {host_api} (Entrée)"))
                if max_output_channels > 0:
                    device_list.append((i, f"{name} - {host_api} (Sortie)"))
            
            return device_list, device_info
            
        except Exception as e:
            print(f"Erreur lors de la liste des périphériques: {e}")
            return [], {}


class InterfaceAnnuleurASIO(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Annuleur de Bruit Pro - Version ASIO Améliorée")
        self.geometry("950x750")
        
        # Thème moderne
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Utiliser un thème plus moderne
        
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
        self.style.configure("Accent.TButton", background=self.couleur_accent)
        self.style.configure("Success.TButton", background=self.couleur_success)
        self.style.configure("Warning.TButton", background=self.couleur_warning)
        
        # Créer l'instance d'annuleur
        self.annuleur = AnuleurDeBruitASIO(taille_tampon=256)  # Taille de tampon réduite pour moins de latence
        
        # Créer l'interface
        self._create_widgets()
        
        # Animation pour la visualisation
        self.ani = None
        self._start_animation()
        
        # Compteur de performances
        self.derniere_maj_perf = time.time()
        self.after(1000, self._update_performances)
        
        # S'assurer que tout est fermé proprement
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _create_widgets(self):
        """Crée tous les widgets de l'interface"""
        # Frame principale
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ===== En-tête avec logo =====
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, padx=5, pady=5)
        
        title_label = ttk.Label(header_frame, text="ANNULEUR DE BRUIT PRO", 
                               font=('Arial', 16, 'bold'), foreground=self.couleur_accent)
        title_label.pack(side=tk.LEFT, padx=5)
        
        version_label = ttk.Label(header_frame, text="v2.0", font=('Arial', 8))
        version_label.pack(side=tk.LEFT)
        
        # Indicateurs de performance
        self.perf_frame = ttk.Frame(header_frame)
        self.perf_frame.pack(side=tk.RIGHT, padx=5)
        
        self.latence_label = ttk.Label(self.perf_frame, text="Latence: -- ms", font=('Arial', 9))
        self.latence_label.pack(side=tk.RIGHT, padx=10)
        
        self.cpu_label = ttk.Label(self.perf_frame, text="CPU: --%", font=('Arial', 9))
        self.cpu_label.pack(side=tk.RIGHT, padx=10)
        
        # ===== Section des périphériques =====
        devices_frame = ttk.LabelFrame(main_frame, text="Périphériques Audio")
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
        
        self.start_button = ttk.Button(buttons_frame, text="▶ Démarrer", command=self._start_processing, style="Success.TButton")
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.stop_button = ttk.Button(buttons_frame, text="⏹ Arrêter", command=self._stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.calib_button = ttk.Button(buttons_frame, text="🔄 Calibrer", command=self._calibrate, style="Accent.TButton")
        self.calib_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Mode de latence
        self.mode_latence_var = tk.BooleanVar(value=True)
        self.mode_latence_check = ttk.Checkbutton(
            buttons_frame, 
            text="Mode faible latence", 
            variable=self.mode_latence_var,
            command=self._toggle_latency_mode
        )
        self.mode_latence_check.pack(side=tk.RIGHT, padx=15, pady=5)
        
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
        
        self.delai_var = tk.DoubleVar(value=5)
        self.scale_delai = ttk.Scale(
            delai_controls, from_=0, to=50, variable=self.delai_var, 
            command=lambda x: self._update_param_label('delai')
        )
        self.scale_delai.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.delai_label = ttk.Label(delai_controls, text="5.0 ms", width=8)
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
        self.ax1.set_xlim(0, 1000)  # Plus de points pour un affichage plus détaillé
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
                    text=f"Traitement en cours. Latence estimée: {sum(self.annuleur.stream.latency)*1000:.1f} ms"
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
        # Convertir délai de ms à échantillons
        delai_echant = int(self.delai_var.get() * self.annuleur.taux_echantillonnage / 1000)
        
        self.annuleur.regler_parametres(
            delai=delai_echant,
            gain=self.gain_var.get(),
            freq_basse=self.freq_basse_var.get(),
            freq_haute=self.freq_haute_var.get(),
            mode_faible_latence=self.mode_latence_var.get()
        )
        
        self.status_label.config(text="Paramètres appliqués.")
        
    def _toggle_latency_mode(self):
        """Bascule entre les modes de latence"""
        if self.mode_latence_var.get():
            # Mode faible latence
            if self.annuleur.en_cours:
                if messagebox.askyesno(
                    "Mode faible latence", 
                    "Le mode faible latence va redémarrer le stream audio avec des paramètres optimisés pour minimiser la latence. Continuer?"
                ):
                    self.annuleur.regler_parametres(mode_faible_latence=True)
                else:
                    self.mode_latence_var.set(False)
        else:
            # Mode qualité
            if self.annuleur.en_cours:
                if messagebox.askyesno(
                    "Mode qualité", 
                    "Le mode qualité va redémarrer le stream audio avec des paramètres optimisés pour une meilleure qualité audio. La latence peut augmenter. Continuer?"
                ):
                    self.annuleur.regler_parametres(
                        mode_faible_latence=False,
                        ordre=4,
                        taille_tampon=512
                    )
                else:
                    self.mode_latence_var.set(True)
        
    def _calibrate(self):
        """Lance la calibration automatique"""
        if not self.annuleur.en_cours:
            messagebox.showinfo("Calibration", "Veuillez d'abord démarrer le traitement avant de calibrer.")
            return
            
        # Lancer la calibration
        try:
            delai, gain = self.annuleur.calibrer()
            
            # Mettre à jour les sliders
            delai_ms = delai * 1000 / self.annuleur.taux_echantillonnage
            self.delai_var.set(delai_ms)
            self.gain_var.set(gain)
            
            # Mettre à jour les labels
            self._update_param_label('delai')
            self._update_param_label('gain')
            
            # Message de réussite avec animation
            self.status_label.config(text="Calibration terminée avec succès !")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Échec de la calibration: {str(e)}")
    
    def _update_performances(self):
        """Met à jour les indicateurs de performances"""
        if self.annuleur.en_cours and hasattr(self.annuleur, 'stream') and self.annuleur.stream is not None:
            # Calculer la latence totale
            latence_totale = sum(self.annuleur.stream.latency) * 1000  # en ms
            
            # Mettre à jour les labels
            self.latence_label.config(text=f"Latence: {latence_totale:.1f} ms")
            self.cpu_label.config(text=f"CPU: {self.annuleur.cpu_usage:.1f}%")
            
            # Colorer selon les performances
            if latence_totale < 10:
                self.latence_label.config(foreground=self.couleur_success)
            elif latence_totale < 20:
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
        if hasattr(self.annuleur, 'donnees_entree') and hasattr(self.annuleur, 'donnees_sortie'):
            self.line1.set_ydata(self.annuleur.donnees_entree)
            self.line2.set_ydata(self.annuleur.donnees_sortie)
            return self.line1, self.line2
        return self.line1, self.line2
    
    def _start_animation(self):
        """Démarre l'animation pour le graphique en temps réel"""
        self.ani = animation.FuncAnimation(
            self.fig, self._update_plot, interval=50,
            blit=True, cache_frame_data=False
        )
    
    def _show_help(self, event=None):
        """Affiche l'aide sur l'utilisation de l'application"""
        help_text = """
        Guide d'utilisation de l'Annuleur de Bruit Pro
        
        1. Sélection des périphériques:
           - Choisissez votre micro comme entrée
           - Choisissez vos haut-parleurs comme sortie
        
        2. Contrôles:
           - Démarrer: lance le traitement audio
           - Arrêter: interrompt le traitement
           - Calibrer: ajuste automatiquement les paramètres
        
        3. Paramètres:
           - Délai: compense la latence du système (5-15ms recommandé)
           - Gain: intensité de l'annulation (0.8-1.0 recommandé)
           - Fréquences: ajuste la plage de fréquences à traiter
        
        4. Conseils:
           - Le mode faible latence réduit le délai mais peut diminuer la qualité
           - Pour de meilleurs résultats, calibrez après chaque démarrage
           - Si vous entendez des oscillations, réduisez le gain
        
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
    print("=== Annuleur de Bruit Pro - Version ASIO Améliorée ===")
    print("Initialisation du système d'annulation de bruit...")
    
    # Lister les périphériques et APIs disponibles
    print("\n===== PÉRIPHÉRIQUES AUDIO DISPONIBLES =====")
    try:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            print(f"[{i}] {device['name']}")
            print(f"    Entrées: {device['max_input_channels']}, Sorties: {device['max_output_channels']}")
            print(f"    Taux d'échantillonnage par défaut: {device['default_samplerate']} Hz")
            print(f"    Latence minimale: {device.get('default_low_input_latency', 0)*1000:.1f}/{device.get('default_low_output_latency', 0)*1000:.1f} ms (in/out)")
            print()
            
        print("===== APIs AUDIO DISPONIBLES =====")
        apis = sd.query_hostapis()
        for i, api in enumerate(apis):
            print(f"[{i}] {api['name']}")
            print(f"    Périphériques: {len(api['devices'])}")
            print(f"    Périph. par défaut: {api['default_input_device']}/{api['default_output_device']}")
            print()
    except Exception as e:
        print(f"Erreur lors de la liste des périphériques: {e}")
    
    # Démarrer l'interface
    app = InterfaceAnnuleurASIO()
    app.mainloop()
