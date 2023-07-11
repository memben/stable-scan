import fire
from view_control import ViewControl
from pcd_io import read_pcd

# best performant image size for SD
WINDOW_WIDTH = 512
WINDOW_HEIGHT = 512
    
class StableScanCLI:
    def run(self, prompt: str, *filenames: str, webui_api: str = "http://127.0.0.1:7860", width: int = WINDOW_WIDTH, height: int = WINDOW_HEIGHT):
        """Run the stablescan viewer with the given files. 
        Navigate to desired camera position and press 'r' to retexture the point cloud.
        Retexturing will cover a 360 degree view of the point cloud.
        
        Args:
            prompt: The prompt to retexture the point cloud
            *filenames: The filenames to load.
            webui_api: The url of the webui api.
            width: The width of the window.
            height: The height of the window.
        """
        pass

    def control(self, *filenames: str, webui_api: str = "http://127.0.0.1:7860", width: int = WINDOW_WIDTH, height: int = WINDOW_HEIGHT):
        """Run the stablescan viewer with the given files. 
        Navigate to desired camera position and press 'r' to retexture the point cloud.
        It will only retexture the points that are visible in the current view.
        Previous generations will be used as a basis for the retexture.
        
        Args:
            *filenames: The filenames to load.
            webui_api: The url of the webui api.
            width: The width of the window.
            height: The height of the window.
        """
        StableScan(*filenames, webui_api=webui_api, width=width, height=height)


    def debug(self, *filenames: str, webui_api: str = "http://127.0.0.1:7860", width: int = WINDOW_WIDTH, height: int = WINDOW_HEIGHT):
        """Run the stablescan viewer with the given files. 
        Extending the capabilities of the control mode.
        Press 'i' to show the indices of the points.
        Press 'f' to show the depth images, filtered and unfiltered, and the effect of the applied filters.
        Press 'x' to show the texture applied to the point cloud without the untexured points.

        
        Args:
            *filenames: The filenames to load.
            webui_api: The url of the webui api.
            width: The width of the window.
            height: The height of the window.
        """
        StableScan(*filenames, webui_api=webui_api, width=width, height=height, debug=True)

class StableScan:
    """High level StableScan instance"""
    def __init__(self, *filenames: str, webui_api: str, width: int, height: int, debug: bool = False):
        self.pcd = read_pcd(*filenames)
        self.vc = ViewControl(self.pcd, width, height, debug)
        self.vc.run()



if __name__ == "__main__":
    fire.Fire(StableScanCLI)