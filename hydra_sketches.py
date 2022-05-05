import asyncio
from skimage import io 
from pyppeteer import launch
from main import count_colors, measure_change, measure_entropy
from urllib.parse import parse_qs
from urllib.parse import urlparse


async def main():
    browser = await launch()
    page = await browser.newPage()
    print("sketch_id,change_score,color_count,entropy_score")
    sketches = set([
        "khoparzi_3",
        "malitzin_1",
        "alexandre_1",
        "mahalia_4",
        "example_4",
        "andromeda_0",
        "example_17",
        "celeste_1",
        "example_13",
        "alexandre_1",
        "example_9",
        "andromeda_1",
        "eerie_ear_0",
        "khoparzi_2",
        "ritchse_3",
        "marianne_1",
        "malitzin_2",
        "malitzin_0",
        "rangga_0",
        "marianne_0",
        "ritchse_0",
        "celeste_2",
        "ritchse_2",
        "asdrubal_0",
        "example_15",
        "mahalia_4",
        "flor_0",
        "marianne_0",
        "example_13",
        "ritchse_4",
        "khoparzi_0",
        "mahalia_3",
    ])
    for sketch_id in sketches:
        await page.goto(f"https://hydra.ojack.xyz?sketch_id={sketch_id}")
        await page.click("i[id=close-icon]")
        #url = page.url
        #sketch_id = parse_qs(urlparse(url).query)["sketch_id"][0]
        images = []
        for j in range(6):
            filename = f"images/hydra_sketch_{sketch_id}_{j}.png"
            await page.screenshot({"path": filename})
            image = io.imread(filename)
            images.append(image)
        change_score = measure_change(images)
        color_count = count_colors(images)
        entropy_score = measure_entropy(images)
        print(f"{sketch_id},{change_score},{color_count},{entropy_score}")
        await page.click("i[id=shuffle-icon]")


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
