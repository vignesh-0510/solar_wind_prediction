import os
import pickle5 as pickle
from typing import Dict, List
import urllib3
import requests
import re
from tqdm import tqdm
import pickle
import wget
from datetime import datetime


class Constants:

    FILES_DICT = {
        # 'vr002': 'helio/vr002.hdf',     # Radial velocity
        # 'br002': 'helio/br002.hdf',     # Radial magnetic field
        # 'vt002': 'helio/vt002.hdf',     # Velocity Theta
        # 'vp002': 'helio/vp002.hdf',     # Velocity Phi
        # 'bt002': 'helio/bt002.hdf',     # Magnetic field Theta
        # 'bp002': 'helio/bp002.hdf',     # Magnetic field Phi
        # 'jt002': 'helio/jt002.hdf',     # Current density Theta
        # 'jp002': 'helio/jp002.hdf',     # Current density Phi
        # 'jr002': 'helio/jr002.hdf',     # Current density Radial
        # 'rho002': 'helio/rho002.hdf',   # Density
        # 'p002': 'helio/p002.hdf',       # Pressure
        # 't002': 'helio/t002.hdf',       # Temperature
        'vr002': 'corona/vr002.hdf',     # Radial velocity
        'br002': 'corona/br002.hdf',     # Radial magnetic field
        'vt002': 'corona/vt002.hdf',     # Velocity Theta
        'vp002': 'corona/vp002.hdf',     # Velocity Phi
        'bt002': 'corona/bt002.hdf',     # Magnetic field Theta
        'bp002': 'corona/bp002.hdf',     # Magnetic field Phi
        'jt002': 'corona/jt002.hdf',     # Current density Theta
        'jp002': 'corona/jp002.hdf',     # Current density Phi
        'jr002': 'corona/jr002.hdf',     # Current density Radial
        'rho002': 'corona/rho002.hdf',   # Density
        'p002': 'corona/p002.hdf',       # Pressure
        't002': 'corona/t002.hdf',       # Temperature
    }  # Simulations of velocity, 140 images


class CheckURLs:
    """
    This class checks available simulations and URLs on www.predsci.com website.
    """

    def __init__(
        self,
        start_dir: int,
        end_dir: int,
        start_url: str = "http://www.predsci.com/data/runs/cr",
        end_url: str = "-medium/",
        save_pickle: bool = False,
    ) -> None:
        self.start_dir = start_dir
        self.end_dir = end_dir
        self.start_url = start_url
        self.end_url = end_url
        self.url_dict = None
        self.cr_num_dict = None
        self.save_pickle = save_pickle
        pass

    def collectURLsAll(self) -> Dict:
        usefulurl = {}
        useful_cr_num = {}
        for i in tqdm(range(self.start_dir, self.end_dir)):
            url = self.start_url + str(i) + self.end_url
            # print('---------------------------------------------')
            # print('URL is: ',url)
            deadlink = self.deadLinkFound(url)
            if not deadlink:
                response = requests.get(url)
                links = re.findall(r'<a[^>]* href="([^"]*)"', response.text, flags=re.IGNORECASE)
                
                # sources = [os.path.join(f'cr{i}{self.end_url.strip("/")}', link) for link in links if link != "../"]
                sources = [link for link in links if link != "../"]
                # print(sources)
                for src in sources:
                    src = src.strip("/")
                    # print('If url exists: True')
                    # print(url)
                    # usefulurl.append(url)  
                    # print(f'cr{i}')
                    usefulurl[src] = self.dict_add(usefulurl, src, url)
                    useful_cr_num[src] = self.dict_add(useful_cr_num, src, f"cr{i}")
            else:
                dummy = []
        self.url_dict = usefulurl
        self.cr_num_dict = useful_cr_num

        if self.save_pickle:
            save_path = "/data/solar_wind_pred_vignesh/"
            filename = (
                "url_dict_" + ".pickle"
            )  # datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
            filename = save_path + filename
            with open(filename, "wb") as handle:
                pickle.dump(self.url_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return self.url_dict

    def deadLinkFound(self, path) -> bool:
        try:
            http = urllib3.PoolManager()
            r = http.request("GET", path)
            response = r.status
            if response == 200:
                return False
            else:
                return True
        except:
            return True

    def dict_add(self, dict_input, key, value) -> List:
        if key not in dict_input:
            return [value]
        else:
            return dict_input[key] + [value]


class DownloadURL:
    """
    This class downloads all URL from given dictionary
    """

    # data/hdf/cr_number/sim_name/files
    def __init__(
        self,
        url: str,
        sim_name: str,
        dir_path: str = "/data/solar_wind_pred_vignesh",
        dir_name: str = "corona/medium",
        file_names_dict: Dict = Constants.FILES_DICT,
    ) -> None:
        self.dir_path = dir_path
        self.dir_name = dir_name
        self.url = url.replace("http://", "https://")
        self.sim_name = sim_name
        self.cr_num = self.url.split("/")[-2].split("-")[0]
        self.filenames = file_names_dict

        self.path = (
            self.dir_path
            + "/"
            + self.dir_name
            + "/"
            + self.cr_num
            + "/"
            + self.sim_name
        )

        os.makedirs(self.path, exist_ok=True)

    def get_simulation(self):
        fn_keys = list(self.filenames.keys())
        fn_path_list = {}
        for idx, fn in enumerate(self.filenames):
            final_url = self.url + self.sim_name + "/" + self.filenames[fn]
            path = self.path  # + '/' + fn_keys[idx]
            wget.download(final_url, path)
            file_name = fn + ".hdf"
            file_name = path + "/" + file_name
            os.rename(path + "/" + fn + ".hdf", file_name)
            fn_path_list[fn] = file_name

        return fn_path_list


def main():
    # total available carrington rotations are [1625, 2240]
    download_ = True
    if download_:
        urls = CheckURLs(1629, 1631, save_pickle=True)
        _ = urls.collectURLsAll()

    #### URLs Downloader ###
    url_dict_path = "/data/solar_wind_pred_vignesh/url_dict_.pickle"
    # loading saved pickle file
    with open(url_dict_path, "rb") as handle:
        urls_dict = pickle.load(handle)
    print(urls_dict)
    # list of simulations which will be downloaded
    dwnld_list = [
        "kpo_mas_mas_std_0101",
        "mdi_mas_mas_std_0101",
        "hmi_mast_mas_std_0101",
        "hmi_mast_mas_std_0201",
        "hmi_masp_mas_std_0201",
        "mdi_mas_mas_std_0201",
    ]

    # creating a new dict of simulations which needs to be downloaded
    top_n_url_dict = {}
    for sim_name in dwnld_list:
        if sim_name in urls_dict:
            top_n_url_dict[sim_name] = urls_dict[sim_name]
    for sim_name in top_n_url_dict.keys():
        url_list = top_n_url_dict[sim_name]
        for url_ in url_list:
            dwnld = DownloadURL(url_, sim_name)
            _ = dwnld.get_simulation()
            cr_num = url_.split("/")[-2].split("-")[0]
            print(f"{sim_name} - {cr_num}")


if __name__ == "__main__":
    main()
