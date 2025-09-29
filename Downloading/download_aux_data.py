import asyncio
import os
import aiofiles
import aiohttp

async def save_file(resp, path):
    async with aiofiles.open(path, mode="wb") as file:
        while True:
            chunk = await resp.content.read(1024)
            if not chunk:
                break
            await file.write(chunk)
    #print(f"Saved file in {path}")
async def get_label(BASE_URL, session, sem, location, time):
    async with sem:
        label_url = f'{BASE_URL}/{location}/{location}_{time}/{location}_{time}-hsi0/labels.bin.tar.gz'
        async with session.get(label_url) as resp:
            path = os.path.join('data', f'{location}_{time}', 'labels.bin.tar.gz')
            await save_file(resp, path)
async def get_longitudes(BASE_URL, session, sem, location, time):
    async with sem:
        longitudes_url = f'{BASE_URL}/{location}/{location}_{time}/processing-temp/longitudes.dat'
        async with session.get(longitudes_url) as resp:
            path = os.path.join('data', f'{location}_{time}', f'{location}_{time}-longitudes.dat')
            await save_file(resp, path)
async def get_latitudes(BASE_URL, session, sem, location, time):
    async with sem:
        latitudes_url = f'{BASE_URL}/{location}/{location}_{time}/processing-temp/latitudes.dat'
        async with session.get(latitudes_url) as resp:
            path = os.path.join('data', f'{location}_{time}', f'{location}_{time}-latitudes.dat')
            await save_file(resp, path)
async def get_capture_config(BASE_URL, session, sem, location, time):
    async with sem:
        capture_config_url = f'{BASE_URL}/{location}/{location}_{time}/{location}_{time}-hsi0/capture_config.ini'
        async with session.get(capture_config_url) as resp:
            path = os.path.join('data', f'{location}_{time}', 'capture_config.ini')
            await save_file(resp, path)
async def get_jon_cnn_labels(BASE_URL, session, sem, location, time):
    async with sem:
        jon_cnn_labels_url = f'{BASE_URL}/{location}/{location}_{time}/processing-temp/jon-cnn.labels'
        #jon_cnn_labels_url = f'{BASE_URL}/{location}/{location}_{time}/{location}_{time}-hsi0/jon-cnn.labels'
        async with session.get(jon_cnn_labels_url) as resp:
            path = os.path.join('data', f'{location}_{time}', 'jon-cnn.labels')
            await save_file(resp, path)
async def get_local_angles(BASE_URL, session, sem, location, time):
    async with sem:
        local_angles_url = f'{BASE_URL}/{location}/{location}_{time}/processing-temp/local-angles.csv'
        async with session.get(local_angles_url) as resp:
            path = os.path.join('data', f'{location}_{time}', f'{location}_{time}-local-angles.csv')
            await save_file(resp, path)
async def get_gcp(BASE_URL, session, sem, location, time):
    async with sem:
        local_angles_url = f'{BASE_URL}/{location}/{location}_{time}/processing-temp/sift-bin.points'
        async with session.get(local_angles_url) as resp:
            path = os.path.join('data', f'{location}_{time}', f'{location}_{time}-sift-bin.points')
            await save_file(resp, path)
async def get_timeposition(BASE_URL, session, sem, location, time):
    async with sem:
        latitudes_url = f'{BASE_URL}/{location}/{location}_{time}/processing-temp/frametime-pose.csv'
        async with session.get(latitudes_url) as resp:
            path = os.path.join('data', f'{location}_{time}', 'frametime-pose.csv')
            await save_file(resp, path)
async def main(files, hypso):
    n_concurrent_downloads = 10
    sem = asyncio.Semaphore(n_concurrent_downloads)
    if hypso == 1:
        BASE_URL = 'http://129.241.2.147:8008/'
    elif hypso == 2:
        BASE_URL = 'http://129.241.2.147:8009/'
    async with aiohttp.ClientSession() as session:
        tasks = []
        for location, time in files:
            directory = os.path.join('data', f'{location}_{time}')
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            tasks.append(get_label(BASE_URL, session, sem, location, time))
            tasks.append(get_longitudes(BASE_URL, session, sem, location, time))
            tasks.append(get_latitudes(BASE_URL, session, sem, location, time))
            tasks.append(get_capture_config(BASE_URL, session, sem, location, time))
            tasks.append(get_jon_cnn_labels(BASE_URL, session, sem, location, time))
            tasks.append(get_timeposition(BASE_URL, session, sem, location, time))
            #tasks.append(get_local_angles(BASE_URL, session, sem, location, time))
            tasks.append(get_gcp(BASE_URL, session, sem, location, time))
            #else:
                #print(f"Directory {directory} already exists, skipping download tasks.")
                # Skip downloading files that already exist in the directory
        await asyncio.gather(*tasks)
def run_downloads3(files, hypso):
    asyncio.run(main(files, hypso))




