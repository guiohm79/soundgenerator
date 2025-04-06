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

class AnuleurDeBruitASIO:
    def __init__(self, 
                 taux_echantillonnage=48000,
                 taille_tampon=1024,
                 device_entree=None,
                 device_sortie=None,
                 api='asio'):
        """
        Initialise le système d'annulation de bruit avec support ASIO
        
        Args:
            taux_echantillonnage: Fréquence d'échantillonnage en Hz
            taille_tampon: Nombre d'échantillons par trame
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
        self.ordre_filtre = 4
        self.freq_coupure_basse = 50  # Hz
        self.freq_coupure_haute = 4000  # Hz
        self.b, self.a = self._creer_filtre()
        
        # État du filtre
        self.z = signal.lfilter_zi(self.b, self.a) * 0
        
        # Tampon pour stocker les échantillons précédents (pour le délai)
        self.tampon_historique = np.zeros(self.taux_echantillonnage)  # 1 seconde d'historique
        self.position_tampon = 0
        
        # Données pour affichage graphique
        self.donnees_entree = np.zeros(100)
        self.donnees_sortie = np.zeros(100)
        
        # File d'attente pour les données audio
        self.queue_audio = queue.Queue(maxsize=20)
        
        # État d'exécution
        self.en_cours = False
        self.thread_traitement = None
        self.stream = None
    
    def _creer_filtre(self):
        """
        Crée le filtre passe-bande
        """
        return signal.butter(
            self.ordre_filtre, 
            [self.freq_coupure_basse/(self.taux_echantillonnage/2), 
             self.freq_coupure_haute/(self.taux_echantillonnage/2)], 
            btype='band'
        )
    
    def callback_audio(self, indata, outdata, frames, time, status):
        """
        Callback appelé par sounddevice pour traiter l'audio
        """
        if status:
            print(f"Statut audio: {status}")
        
        # Extraire les données audio mono (prendre la première colonne si stéréo)
        donnees_audio = indata[:, 0] if indata.ndim > 1 else indata[:]
        
        # Stocker pour visualisation
        self.donnees_entree = np.roll(self.donnees_entree, -len(donnees_audio))
        if len(donnees_audio) < len(self.donnees_entree):
            self.donnees_entree[-len(donnees_audio):] = donnees_audio
        else:
            self.donnees_entree = donnees_audio[-len(self.donnees_entree):]
        
        # Stocker les données dans le tampon d'historique
        debut_position = self.position_tampon
        fin_position = (debut_position + len(donnees_audio)) % len(self.tampon_historique)
        
        if fin_position > debut_position:
            self.tampon_historique[debut_position:fin_position] = donnees_audio
        else:
            # Gestion du cas où le tampon circule
            premier_segment = len(self.tampon_historique) - debut_position
            if premier_segment < len(donnees_audio):
                self.tampon_historique[debut_position:] = donnees_audio[:premier_segment]
                self.tampon_historique[:fin_position] = donnees_audio[premier_segment:premier_segment+fin_position]
            else:
                self.tampon_historique[debut_position:debut_position+len(donnees_audio)] = donnees_audio
        
        self.position_tampon = fin_position
        
        # Appliquer un filtre passe-bande
        donnees_filtrees, self.z = signal.lfilter(self.b, self.a, donnees_audio, zi=self.z)
        
        # Inverser la phase (multiplier par -1)
        donnees_inversees = -self.gain * donnees_filtrees
        
        # Compensation de délai si nécessaire
        if self.delai_compensation > 0:
            position_delai = (self.position_tampon - self.delai_compensation) % len(self.tampon_historique)
            donnees_retardees = np.zeros_like(donnees_inversees)
            
            for i in range(len(donnees_inversees)):
                idx = (position_delai + i) % len(self.tampon_historique)
                donnees_retardees[i] = self.tampon_historique[idx]
            
            # Mélanger le signal retardé avec le signal inversé
            donnees_sortie = donnees_retardees + donnees_inversees
        else:
            donnees_sortie = donnees_inversees
        
        # Stocker pour visualisation
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
                print(f"Latence d'entrée: {self.stream.latency[0]:.2f}s, Latence de sortie: {self.stream.latency[1]:.2f}s")
                
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
        
        # Utilisez la latence reportée par le stream pour estimer le délai
        if self.stream and hasattr(self.stream, 'latency'):
            latence_totale = sum(self.stream.latency)
            self.delai_compensation = int(latence_totale * self.taux_echantillonnage)
        else:
            # Valeur par défaut si la latence n'est pas disponible
            self.delai_compensation = int(0.005 * self.taux_echantillonnage)  # 5ms de délai
            
        self.gain = 0.95  # Légèrement réduit pour éviter les oscillations
        
        print(f"Calibration terminée: délai={self.delai_compensation} échantillons ({self.delai_compensation/self.taux_echantillonnage*1000:.1f}ms), gain={self.gain}")
        return self.delai_compensation, self.gain
        
    def regler_parametres(self, delai=None, gain=None, freq_basse=None, freq_haute=None, 
                          ordre=None, taux_echant=None, taille_tampon=None):
        """
        Permet de régler manuellement les paramètres
        """
        recalculer_filtre = False
        redemarrer_stream = False
        
        if delai is not None:
            self.delai_compensation = delai
        if gain is not None:
            self.gain = gain
        if ordre is not None:
            self.ordre_filtre = ordre
            recalculer_filtre = True
        if taux_echant is not None and taux_echant != self.taux_echantillonnage:
            self.taux_echantillonnage = taux_echant
            recalculer_filtre = True
            redemarrer_stream = True
        if taille_tampon is not None and taille_tampon != self.taille_tampon:
            self.taille_tampon = taille_tampon
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
                    'outputs': max_output_channels
                }
                
                # Ajouter à la liste de sélection
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
        
        self.title("Annuleur de Bruit Pro - Version ASIO")
        self.geometry("900x700")
        self.configure(bg="#f0f0f0")
        
        # Créer l'instance d'annuleur
        self.annuleur = AnuleurDeBruitASIO()
        
        # Créer l'interface
        self._create_widgets()
        
        # Animation pour la visualisation
        self.ani = None
        self._start_animation()
        
        # S'assurer que tout est fermé proprement
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _create_widgets(self):
        """Crée tous les widgets de l'interface"""
        # Frame principale
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Style pour les widgets
        style = ttk.Style()
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabel", background="#f0f0f0", font=('Arial', 10))
        style.configure("TButton", font=('Arial', 10))
        style.configure("Header.TLabel", font=('Arial', 12, 'bold'))
        
        # ===== Section des périphériques =====
        devices_frame = ttk.LabelFrame(main_frame, text="Périphériques Audio ASIO")
        devices_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Récupérer la liste des périphériques
        self.devices, self.device_info = self.annuleur.liste_peripheriques()
        
        # Entrée audio
        ttk.Label(devices_frame, text="Entrée audio:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.entree_var = tk.StringVar()
        self.combo_entree = ttk.Combobox(devices_frame, textvariable=self.entree_var, state="readonly", width=40)
        self.combo_entree['values'] = [d[1] for d in self.devices if "Entrée" in d[1]]
        self.combo_entree.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        if self.combo_entree['values']:
            self.combo_entree.current(0)
        
        # Sortie audio
        ttk.Label(devices_frame, text="Sortie audio:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.sortie_var = tk.StringVar()
        self.combo_sortie = ttk.Combobox(devices_frame, textvariable=self.sortie_var, state="readonly", width=40)
        self.combo_sortie['values'] = [d[1] for d in self.devices if "Sortie" in d[1]]
        self.combo_sortie.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        if self.combo_sortie['values']:
            self.combo_sortie.current(0)
        
        # ===== Section des contrôles =====
        controls_frame = ttk.LabelFrame(main_frame, text="Contrôles")
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Boutons
        buttons_frame = ttk.Frame(controls_frame)
        buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_button = ttk.Button(buttons_frame, text="Démarrer", command=self._start_processing)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.stop_button = ttk.Button(buttons_frame, text="Arrêter", command=self._stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.calib_button = ttk.Button(buttons_frame, text="Calibrer", command=self._calibrate)
        self.calib_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # ===== Section des paramètres =====
        params_frame = ttk.LabelFrame(main_frame, text="Paramètres")
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Délai
        ttk.Label(params_frame, text="Délai (ms):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.delai_var = tk.DoubleVar(value=5)
        self.scale_delai = ttk.Scale(params_frame, from_=0, to=50, variable=self.delai_var, 
                                     command=lambda x: self._update_param_label('delai'))
        self.scale_delai.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        self.delai_label = ttk.Label(params_frame, text="5.0 ms")
        self.delai_label.grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        
        # Gain
        ttk.Label(params_frame, text="Gain:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.gain_var = tk.DoubleVar(value=0.95)
        self.scale_gain = ttk.Scale(params_frame, from_=0, to=2, variable=self.gain_var, 
                                    command=lambda x: self._update_param_label('gain'))
        self.scale_gain.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        self.gain_label = ttk.Label(params_frame, text="0.95")
        self.gain_label.grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        
        # Fréquence basse
        ttk.Label(params_frame, text="Fréq. basse (Hz):").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.freq_basse_var = tk.DoubleVar(value=50)
        self.scale_freq_basse = ttk.Scale(params_frame, from_=20, to=1000, variable=self.freq_basse_var, 
                                         command=lambda x: self._update_param_label('freq_basse'))
        self.scale_freq_basse.grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)
        self.freq_basse_label = ttk.Label(params_frame, text="50 Hz")
        self.freq_basse_label.grid(row=2, column=2, padx=5, pady=5, sticky=tk.W)
        
        # Fréquence haute
        ttk.Label(params_frame, text="Fréq. haute (Hz):").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.freq_haute_var = tk.DoubleVar(value=4000)
        self.scale_freq_haute = ttk.Scale(params_frame, from_=1000, to=20000, variable=self.freq_haute_var, 
                                         command=lambda x: self._update_param_label('freq_haute'))
        self.scale_freq_haute.grid(row=3, column=1, padx=5, pady=5, sticky=tk.EW)
        self.freq_haute_label = ttk.Label(params_frame, text="4000 Hz")
        self.freq_haute_label.grid(row=3, column=2, padx=5, pady=5, sticky=tk.W)
        
        # Appliquer les changements
        self.apply_button = ttk.Button(params_frame, text="Appliquer", command=self._apply_params)
        self.apply_button.grid(row=4, column=1, padx=5, pady=5)
        
        # ===== Information sur l'API audio =====
        api_frame = ttk.Frame(params_frame)
        api_frame.grid(row=5, column=0, columnspan=3, padx=5, pady=5, sticky=tk.W)
        
        try:
            api_info = sd.query_hostapis()
            available_apis = [api['name'] for api in api_info]
            api_text = f"APIs audio disponibles: {', '.join(available_apis)}"
            
            # Trouver l'API ASIO
            asio_available = "ASIO" in ''.join(available_apis)
            if asio_available:
                api_text += "\nASIO est disponible ! 🎉"
            else:
                api_text += "\nASIO n'est pas détecté. Vérifiez vos pilotes."
                
            ttk.Label(api_frame, text=api_text, wraplength=400).pack(anchor=tk.W)
            
        except Exception as e:
            ttk.Label(api_frame, text=f"Erreur lors de la vérification des APIs: {e}", 
                     wraplength=400).pack(anchor=tk.W)
        
        # ===== Section de visualisation =====
        viz_frame = ttk.LabelFrame(main_frame, text="Visualisation des signaux")
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Créer la figure matplotlib
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6), dpi=80)
        self.fig.tight_layout(pad=3.0)
        
        # Configurer les axes
        self.ax1.set_title("Signal d'entrée")
        self.ax1.set_ylim(-1, 1)
        self.ax1.set_xlim(0, 100)
        self.ax1.grid(True)
        
        self.ax2.set_title("Signal de sortie (inversé)")
        self.ax2.set_ylim(-1, 1)
        self.ax2.set_xlim(0, 100)
        self.ax2.grid(True)
        
        # Line plots
        x = np.arange(100)
        self.line1, = self.ax1.plot(x, np.zeros(100), 'r-', lw=1)
        self.line2, = self.ax2.plot(x, np.zeros(100), 'b-', lw=1)
        
        # Intégrer la figure dans tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configurer les colonnes pour qu'elles s'étendent
        params_frame.columnconfigure(1, weight=1)
        
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
    
    def _apply_params(self):
        """Applique les paramètres actuels"""
        # Convertir délai de ms à échantillons
        delai_echant = int(self.delai_var.get() * self.annuleur.taux_echantillonnage / 1000)
        
        self.annuleur.regler_parametres(
            delai=delai_echant,
            gain=self.gain_var.get(),
            freq_basse=self.freq_basse_var.get(),
            freq_haute=self.freq_haute_var.get()
        )
        
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
            
            messagebox.showinfo("Calibration", "Calibration terminée avec succès !")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Échec de la calibration: {str(e)}")
    
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
    print("Initialisation de l'annuleur de bruit avec support ASIO...")
    
    # Lister les périphériques et APIs disponibles
    print("===== PÉRIPHÉRIQUES AUDIO DISPONIBLES =====")
    try:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            print(f"[{i}] {device['name']}")
            print(f"    Entrées: {device['max_input_channels']}, Sorties: {device['max_output_channels']}")
            print(f"    Taux d'échantillonnage par défaut: {device['default_samplerate']} Hz")
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