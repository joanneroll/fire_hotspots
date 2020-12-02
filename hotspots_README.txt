FIRE HOTSPOTS - Python Skripte

Anwendung
- die Pythonskripte muessen in den Ordner kopiert werden mit
	- Hotspotdaten im Format 'gebiet_zeit.gpgk'
	- Untersuchungsgebiet im .shp Format
- Ausfuehren der Skripte in Prompt
- Getestet fuer Windows 

Mit den folgenden Skripten, koennen die Fire Hotspot Daten analysiert werden:
hotspots_clip_data.py: 
- Genaues Zuschneiden der Daten auf Untersuchungsgebiet 
- Ergebnis in: 
	- homedirectory --> data_clipped

hotspots_stats_plotting.py:
- Statistische Analyse der Daten pro verfuegbaren Zeitschritt hinsichtlich: 
	- Anzahl von Detektionen pro Polygon
	- Durschnittsintensitaet (avgFRP) der Hotspots pro Polygon
	- Start und Ende der Satellitendetektion von Hotspoten in Polygon
- Speichern der statistischen Analyse in Excel-Sheets
- Plotten der verfuegbaren Zeitschritte: zwei thematische Karten
	- Anzahl von Detektionen pro Polygon
	- Durchschnittsintensitaet der Hotspots 
	- Symbologie farblich von Verteilung der Daten ueber die gesamte Zeitreihe abgeleitet
- Ergebnis in: 
	- homedirectory --> data_clipped --> output --> statistics
	- homedirectory --> data_clipped --> output --> plotting_data

hotspots_spatial_clustering.py:
- Raeumliches Clustern der Daten mit zwei Algorithmen:
	- KMEANS: Visualisierung der raeumlichen Verteilung der Daten 
	- DBSCAN: Visualisierung von Haeufungen der Daten
- Visualisierung der Cluster in Karten
- Details der Clusterergebnisse als .txt Datei
- Exportieren der Daten mit Clusterergebnissen als neuen Attributen
- Ergebnis in: 
	- homedirectory --> data_clipped --> output --> spatial_clustering

hotspots_clustering_det_int.py:
- Clustern der Daten mit KMEANS nach:
	- Anzahl von Detektionen pro Polygon
	- Durschnittsintensitaet (avgFRP) der Hotspots pro Polygon
- Visualisierung der Cluster als:
	- Scatterplot
	- Karte
- Details der Clusterergebnisse als .txt Datei
- Exportieren der Daten mit Clusterergebnissen als neuen Attributen
- Ergebnis in: 
	- homedirectory --> data_clipped --> output --> clustering_det_int

hotspots_clustering_det_int_time.py:
- Visualisierung der Daten als Scatterplot hinsichtlich
	- Anzahl von Detektionen pro Polygon
	- Durschnittsintensitaet (avgFRP) der Hotspots pro Polygon
	- Zeitpunkt der Detektion (Mean aus tstart und tend) 
- Clustern der Daten mit KMEANS (zur besseren Visualisierung)
- Ergebnis in: 
	- homedirectory --> data_clipped --> output --> clustering_det_int_time
