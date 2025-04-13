
# ======================== overlay_map.py ========================
import folium
import webbrowser
import tkinter as tk
from tkintermapview import TkinterMapView

initial_coords = (13.736717, 100.523186)  # Bangkok

def get_initial_coords():
    root = tk.Tk()
    root.title("เลือกจุดเริ่มต้นของโดรน")
    map_widget = TkinterMapView(root, width=800, height=600)
    map_widget.pack(fill="both", expand=True)
    map_widget.set_position(*initial_coords)
    map_widget.set_zoom(14)

    coords = []

    def set_marker(marker):
        coords.clear()
        coords.extend(marker.position)
        root.destroy()

    map_widget.add_right_click_menu_command(label="ตั้งจุดเริ่มต้น", command=set_marker, pass_coords=True)
    root.mainloop()

    return coords if coords else initial_coords