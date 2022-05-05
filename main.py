import subprocess
import hashlib
import asyncio
import numpy as np
from generator import CodeGenerator
from pyppeteer import launch
import esprima
import random
from numpy.random import rand
import jscodegen
from genetic import genetic_algorithm

from skimage import io 
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from scipy.stats import burr

from sklearn import metrics


async def render_code(page, code):
    await page.goto(f"https://hydra.ojack.xyz/?code={code}")
    images = []
    for i in range(4):
        filename = f"images/{hashlib.md5(code.encode('utf-8')).hexdigest()}_{i}.png"
        await page.screenshot({"path": filename})
        image = io.imread(filename)
        images.append(image)
    return images


def mutual_information(img1, img2, bins=10):
     """ 
     Mutual information for joint histogram
     From https://matthew-brett.github.io/teaching/mutual_information.html
     """
     hgram = np.histogram2d(img1.ravel(), img2.ravel(), bins=20)[0]
     # Convert bins counts to probability values
     pxy = hgram / float(np.sum(hgram))
     px = np.sum(pxy, axis=1) # marginal for x over y
     py = np.sum(pxy, axis=0) # marginal for y over x
     px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
     # Now we can do the calculation using the pxy, px_py 2D arrays
     nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
     return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def measure_entropy(images):
    ents = []
    for image in images:
        ent = entropy(img_as_ubyte(rgb2gray(image[:,:,:3])), disk(7))
        ents.append(np.average(ent))
    return sum(ents) / len(ents)

def measure_change(images):
    change_amounts = []
    for i in range(0, len(images), 2):
        img1, img2 = images[i], images[i+1]
        change_amount = 1 / mutual_information(img1, img2)
        change_amounts.append(change_amount)
    return np.average(change_amounts)

def count_colors(images):
    colors = set()
    for img in images:
        colors = colors.union(set(tuple(v) for m2d in img for v in m2d))
    return len(colors)

def measure_appeal(images):
    change_amount = measure_change(images)
    # values dervived from distributions.ipynb
    change_score = burr.pdf(change_amount, 2.7964741676501434, 107.29523872628266, -0.660537175744617, 0.3348659583417909)
    print(f"change score: {change_score}")

    color_count = count_colors(images)
    color_score = burr.pdf(color_count, 0.5797647610212682, 7.18655286355451, -1.9248129339763533, 1940.106362851806)
    print(f"color score: {color_score}")

    entropy = measure_entropy(images)
    entropy_score = burr.pdf(entropy, 15.567727791355328, 0.2384381159550322, -1.9761150845550512, 7.310259342526781)
    print(f"entropy score: {entropy_score}")

    # Apply weights to get each to a max of 1.2
    return (change_score * 4) + (color_score * 1e5) + (entropy_score * 2)


async def make_fitness_func():
    browser = await launch(args=["--size=600,600"])
    page = await browser.newPage()
    async def fitness(code):
        gen = CodeGenerator()
        encoded = gen.encodeText(code)
        images = await render_code(page, encoded)
        appeal = measure_appeal(images)
        return appeal
    return fitness, browser, page

def find_sources(code):
    result = []
    class Visitor(esprima.NodeVisitor):
        def transform_CallExpression(self, node, metadata):
            if node.callee.name in CodeGenerator.sourcesList:
                result.append(node)
            return self.generic_visit(node)
    visitor = Visitor()
    tree = esprima.parseScript(code, delegate=visitor)
    visitor.visit(tree)
    return result


def run_prettier(code):
    try:
        p = subprocess.Popen( ["npx", "prettier", "--stdin-filepath", "fake.js"], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE,)
        result = p.communicate(input=code.encode())[0].decode()
        if not result:
            print("Warning: could not make code pretty")
            return code
        return result
    except subprocess.CalledProcessError as e:
        print("Warning: could not make code pretty")
        print(e.output)
        return code


def random_replace(code, sources):
    class Visitor(esprima.NodeVisitor):
        def transform_CallExpression(self, node, metadata):
            new_node = node
            if node.callee.name in CodeGenerator.sourcesList:
                new_node = random.choice(sources)
            return self.generic_visit(new_node)
    visitor = Visitor()
    tree = esprima.parseScript(code, delegate=visitor)
    visitor.visit(tree)
    result = CodeGenerator.info + jscodegen.generate(tree.toDict())
    result = run_prettier(result)
    return result

def crossover(parent1, parent2, r_cross):
    sources1 = find_sources(parent1)
    sources2 = find_sources(parent2)
    child1 = random_replace(parent1, sources2)
    child2 = random_replace(parent2, sources1)
    return [child1, child2]

def mutation(code, r_mut):
    # TODO: maybe just tweak literal values here like the editor does
#    class Visitor(esprima.NodeVisitor):
#        def transform_CallExpression(self, node, metadata):
#            new_node = node
#            if rand() > r_mut and new_node.callee.name in CodeGenerator.sourcesList:
#                new_node.callee.name = random.choice(CodeGenerator.sourcesList)
#            return self.generic_visit(new_node)
#    visitor = Visitor()
#    tree = esprima.parseScript(code, delegate=visitor)
#    visitor.visit(tree)
    return code


async def main():
    # define the total iterations
    n_iter = 5
    # define the population size
    n_pop = 10
    # crossover rate
    r_cross = 0.9
    # mutation rate
    r_mut = 1.0 / 100.0
    # perform the genetic algorithm search
    fitness_function = (await make_fitness_func())[0]
    browser = await launch(
        headless=False,
        executablePath="/usr/bin/google-chrome",
        args=["--start-fullscreen", "--window-size=2048,1152"],
        defaultViewport={
            "width": 0,
            "height": 0,
        }
    )
    page = await browser.newPage()
    gen = CodeGenerator()
    while True:
        async for best, score in genetic_algorithm(
            creation=lambda: gen.generateCode(minFunctions=3, maxFunctions=10),
            fitness=fitness_function,
            mutation=mutation,
            crossover=crossover,
            n_iter=n_iter,
            n_pop=n_pop,
            r_cross=r_cross,
            r_mut=r_mut,
        ):
            print(f"Best from generation with score: {score}")
            print(best)
            await page.goto(f"https://hydra.ojack.xyz/?code={gen.encodeText(best)}")
        print(f"Final best with score: {score}")
        print(best)
        print("Starting new run")

    await browser.close()

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
