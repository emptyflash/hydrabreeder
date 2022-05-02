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


async def render_code(page, code):
    await page.goto(f"https://hydra.ojack.xyz/?code={code}")
    #await asyncio.sleep(1) # Wait for the hydra logo to go away
    images = []
    for i in range(4):
        filename = f"images/{hashlib.md5(code.encode('utf-8')).hexdigest()}_{i}.png"
        await page.screenshot({"path": filename})
        image = io.imread(filename)
        images.append(image)
    return images


def measure_entropy(images):
    ents = []
    for image in images:
        ent = entropy(rgb2gray(image[:,:,:3]), disk(7))
        ents.append(np.average(ent))
    return sum(ents) / len(ents)

async def make_fitness_func():
    browser = await launch(args=["--size=600,600"])
    page = await browser.newPage()
    async def fitness(code):
        gen = CodeGenerator()
        encoded = gen.encodeText(code)
        images = await render_code(page, encoded)
        ent = measure_entropy(images)
        return ent
    return fitness, browser

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
    return jscodegen.generate(tree.toDict())

def crossover(parent1, parent2, r_cross):
    sources1 = find_sources(parent1)
    sources2 = find_sources(parent2)
    child1 = random_replace(parent1, sources2)
    child2 = random_replace(parent2, sources1)
    return [child1, child2]

def mutation(code, r_mut):
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
    gen = CodeGenerator()
    fitness_function, browser = await make_fitness_func()
    # define the total iterations
    n_iter = 5
    # define the population size
    n_pop = 10
    # crossover rate
    r_cross = 0.9
    # mutation rate
    r_mut = 1.0 / 100.0
    # perform the genetic algorithm search
    browser = await launch(
        headless=False,
        executablePath="/usr/bin/google-chrome",
        args=["--start-fullscreen", "--window-size=2048,1152"],
        defaultViewport={
            "width": 2048,
            "height": 1152
        }
    )
    page = await browser.newPage()
    while True:
        async for best, score in genetic_algorithm( creation=lambda: gen.generateCode(minFunctions=3, maxFunctions=10),
            fitness=fitness_function,
            mutation=mutation,
            crossover=crossover,
            n_iter=n_iter,
            n_pop=n_pop,
            r_cross=r_cross,
            r_mut=r_mut,
        ):
            print(f"New best with entropy: {score}")
            print(best)
            await page.goto(f"https://hydra.ojack.xyz/?code={gen.encodeText(best)}")

    await browser.close()

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
