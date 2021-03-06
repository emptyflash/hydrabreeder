# Ale Cominotti - 2020

import time
import random
import math
import operator
import base64

RED = '\033[91m'
WHITE = '\033[0m'   

# Class that generates sources and functions in Hydra sintax
class CodeGenerator():

    # This info will be on top of the code.
    info=""

    minValue = 0  # lower bound value to set as function argument
    maxValue = 5  # upper bound value to set as function argument
    arrowFunctionProb = 10 # Probabilities of generating an arrow function that changes value over time (ex.: () => Math.sin(time * 0.3))
    mouseFunctionProb = 0 # Probabilities of generating an arrow function that uses mouse position (ex.: () => mouse.x)
    modulateItselfProb = 20 # Probabilities of generating a modulation function with "o0" as argument (ex.: modulate(o0,1))

    mathFunctions = ["sin", "cos", "tan"]
    mouseList = ["mouse.x", "mouse.y"]
    sourcesList = ["gradient", "noise", "osc", "shape", "solid", "voronoi"]
    colorList = ["brightness", "contrast", "color", "colorama", "invert", "luma", "posterize", "saturate", "thresh"]
    geometryList = ["kaleid", "pixelate", "repeat", "repeatX", "repeatY", "rotate", "scale", "scrollX", "scrollY"]
    modulatorsList = ["modulate", "modulateHue", "modulateKaleid", "modulatePixelate", "modulateRepeat", "modulateRepeatX", "modulateRepeatY", "modulateRotate", "modulateScale", "modulateScrollX", "modulateScrollY"]
    operatorsList = ["add", "blend", "diff", "layer", "mask", "mult"]
    functionsList = ["genColor", "genGeometry", "genModulator", "genOperator"]
    ignoredList = ["solid", "brightness", "luma", "invert", "posterize", "thresh", "layer", "modulateScrollX", "modulateScrollY"]
    exclusiveSourceList = []
    exclusiveFunctionList = []

    def __str__(self):
        return "Code Generator Class"
    
    def __init__(self, min=3, max=10, arrowFunctionProb=10, mouseFunctionProb=0, modulateItselfProb=20, ignoredList=None, exclusiveSourceList=None, exclusiveFunctionList=None):
        if not (min is None):
            self.minValue = min
        if not (max is None):
            self.maxValue = max
        if not (min is None) and not (max is None) and (min>max):
            self.printError("Argument max value must be bigger than min value.")
            exit(1)
        if not (ignoredList is None) and (len(ignoredList)>0):
            self.ignoredList = ignoredList
        if not (arrowFunctionProb is None):
            if(0 <= arrowFunctionProb <= 100):
                self.arrowFunctionProb = arrowFunctionProb
            else:
                self.printError("Arrow function probability must be a number between 0 and 100.")
                exit(1)
        if not (mouseFunctionProb is None):
            if(0 <= mouseFunctionProb <= 100):
                self.mouseFunctionProb = mouseFunctionProb
            else:
                self.printError("Mouse arrow function probability must be a number between 0 and 100.")
                exit(1)        
        if not (modulateItselfProb is None):
            if(0 <= modulateItselfProb <= 100):
                self.modulateItselfProb = modulateItselfProb
            else:
                self.printError("Modulate itself probability must be a number between 0 and 100.")
                exit(1)
        if not (exclusiveSourceList is None) and (len(exclusiveSourceList)>0):
            if(self.checkSources(exclusiveSourceList)):
                self.exclusiveSourceList = exclusiveSourceList
            elif not exclusiveSourceList==['']:
                self.printError("One or more of the specified exclusive sources don't exist")
                exit(1)
        if not (exclusiveFunctionList is None) and (len(exclusiveFunctionList)>0):
            if(self.checkFunctions(exclusiveFunctionList)):
                self.exclusiveFunctionList = exclusiveFunctionList
            elif not exclusiveFunctionList==['']:
                self.printError("One or more of the specified exclusive functions don't exist")
                exit(1)
        if(len(self.ignoredList)>0 and (len(self.exclusiveSourceList)>0 or len(self.exclusiveFunctionList)>0)):
            exclusiveSourceAndFunction= self.exclusiveSourceList+self.exclusiveFunctionList
            if( len([i for i in exclusiveSourceAndFunction if i in self.ignoredList]) > 0):
                self.printError("You can't ignore sources or functions specified as exclusive")
                exit(1)
    #GETTERS
    def getInfo(self):
        return self.info
    
    def getMinValue(self):
        return self.minValue

    def getMaxValue(self):
        return self.maxValue
    
    def getArrowFunctionProb(self):
        return self.arrowFunctionProb
    
    def getMouseFunctionProb(self):
        return self.mouseFunctionProb
    
    def getModulateItselfProb(self):
        return self.modulateItselfProb
    
    def getSourcesList(self):
        return self.sourcesList

    def getAllFunctions(self):
        return self.colorList + self.geometryList + self.modulatorsList + self.operatorsList

    def getColorList(self):
        return self.colorList
    
    def getgeometryList(self):
        return self.geometryList

    def getModulatorsList(self):
        return self.modulatorsList

    def getOperatorsList(self):
        return self.operatorsList

    def getFunctionsList(self):
        return self.functionsList

    def getIgnoredList(self):
        return self.ignoredList

    def getExclusiveSourceList(self):
        return self.exclusiveSourceList

    def getExclusiveFunctionList(self):
        return self.exclusiveFunctionList  
    
    def setDriver(self, value):
        self.driver=value
    
    def getDriver(self):
        return self.driver

    #END GETTERS

    def truncate(self, number, digits) -> float:
        stepper = 10.0 ** digits
        return math.trunc(stepper * number) / stepper

    def isIgnored(self, chosen):
        return(chosen.lower() in [x.lower() for x in self.ignoredList])     
    
    def isExclusiveSource(self, chosen):
        if(len(self.exclusiveSourceList)==0):
            return True
        else:
            return(chosen.lower() in [x.lower() for x in self.exclusiveSourceList])     
    
    def isExclusiveFunction(self, chosen):
        if(len(self.exclusiveFunctionList)==0):
            return True
        else:
            return(chosen.lower() in [x.lower() for x in self.exclusiveFunctionList])   

    def checkSources(self, inputSourcesList):
        return set(inputSourcesList).issubset(self.sourcesList)
    
    def checkFunctions(self, inputFunctionsList):
        allFunctions= self.colorList+self.geometryList+self.modulatorsList+self.operatorsList
        return set(inputFunctionsList).issubset(allFunctions)

    def printError(self, message):
        print(RED + "\nERROR: " + WHITE + message)



    # VALUE GENERATION METHODS ---

    def genNormalValue(self):
        randomTruncate = random.randint(0, 3)
        val = self.truncate(random.uniform(self.minValue, self.maxValue), randomTruncate)
        return(str(val))
    
    def genArrowFunctionValue(self):
        randomTimeMultiplier = self.truncate(random.uniform(0.1, 1), random.randint(1,2))
        # probabilities of generating an arrow function
        if(random.randint(1, 100) <= self.arrowFunctionProb):
            return("""() => Math."""+self.mathFunctions[random.randint(0, len(self.mathFunctions)-1)]+"(time * "+str(randomTimeMultiplier)+")")
        # probabilities of generating a mouse function
        if(random.randint(1, 100) <= self.mouseFunctionProb):
            return("""() => """ + self.mouseList[random.randint(0, len(self.mouseList)-1)] + " * " + str(randomTimeMultiplier))
        return("")


    def genValue(self):  # generates a number, mouse, or math functions
        arrow=self.genArrowFunctionValue()
        if(arrow!=""):
            return arrow
        else:
            return self.genNormalValue()

    def genPosOrNegValue(self): # generates a normal number with 1/5 posibilities of being negative
        arrow=self.genArrowFunctionValue()
        if(arrow!=""):
            return arrow
        elif(random.randint(1, 5) == 5):
            return("-" + self.genNormalValue())
        else:
            return(self.genNormalValue())

    def genCeroOneValue(self):  # generates a number between 0 and 1
        arrow=self.genArrowFunctionValue()
        if(arrow!=""):
            return arrow
        else:
            return str(self.truncate(random.uniform(0, 1), 1))
        

    def genCeroPointFiveValue(self):  # generates a number between 0 and 0.5
        arrow=self.genArrowFunctionValue()
        if(arrow!=""):
            return arrow
        else:
            return str(self.truncate(random.uniform(0, 0.5), 2))
    
    def genCeroPointOneToMax(self):  # generates a number between 0.1 and maxValue
        arrow=self.genArrowFunctionValue()
        if(arrow!=""):
            return arrow
        else:
            return str(self.truncate(random.uniform(0.1, self.maxValue), 2))

    def genCeroPointOneToOne(self):  # generates a number between 0.1 and maxValue
        arrow=self.genArrowFunctionValue()
        if(arrow!=""):
            return arrow
        else:
            return str(self.truncate(random.uniform(0.1, 1), 2))        

    # END VALUE GENERATION METHODS ---


    # MAIN METHODS ---

    def generateCode(self, minFunctions, maxFunctions):
        functionsAmount= random.randint(minFunctions, maxFunctions)
        code = ""
        code += self.info
        code += self.genSource() + "\n"
        for x in range(functionsAmount):
            code += '  ' + self.genFunction() + "\n"
        code += ".out(o0)"
        return code


    def genSource(self):  # returns a source calling one of them randomly
        fullSource = operator.methodcaller(random.choice((self.sourcesList)))(self)
        source=fullSource.split("(")[0] # just source name
        start = time.time() # avoids failing when everything is ignored
        while((not self.isExclusiveSource(source)) or self.isIgnored(source) and (time.time() < (start + 10))):
            fullSource = operator.methodcaller(random.choice((self.sourcesList)))(self)
            source=fullSource.split("(")[0]
        if(time.time() >= (start + 15)):
            self.printError("Could't generate a Source (You ignored all of them")
            exit(1)
        else:
            return fullSource

    def genFunction(self):  # returns a function calling one of them randomly
        fullFunction = operator.methodcaller(random.choice((self.functionsList)))(self)
        function = fullFunction[1:].split("(")[0] # just its name
        start = time.time() # avoids failing when everything is ignored
        while((not self.isExclusiveFunction(function)) or (self.isIgnored(function)) and (time.time() < (start + 10)) ):
            fullFunction = operator.methodcaller(random.choice((self.functionsList)))(self)
            function = fullFunction[1:].split("(")[0]
        if(time.time() >= (start + 15)):
            print(RED + "\nERROR:" + WHITE + " Could't generate a Function (You ignored all of them)")
            exit(1)
        else:
            return fullFunction

    # END MAIN METHODS ---


    # FUNCTION METHODS ---

    def genColor(self):  # returns a color function calling one of them randomly
        return operator.methodcaller(random.choice((self.colorList)))(self)

    def genGeometry(self):  # returns a geometry function calling one of them randomly
        return operator.methodcaller(random.choice((self.geometryList)))(self)

    def genModulator(self):  # returns a geometry function calling one of them randomly
        return operator.methodcaller(random.choice((self.modulatorsList)))(self)

    def genOperator(self):  # returns an operator function calling one of them randomly
        return operator.methodcaller(random.choice((self.operatorsList)))(self)

    # END FUNCTION METHODS ---


    # SOURCES ---

    def gradient(self):
        return("gradient("+self.genValue()+")")

    def noise(self):
        return("noise("+self.genValue()+", "+self.genValue()+")")

    def osc(self):
        return("osc("+self.genValue()+", "+self.genValue()+", "+self.genValue()+")")

    def shape(self):
        return("shape("+self.genValue()+", "+self.genCeroPointFiveValue()+", "+self.genCeroPointOneToOne()+")")

    def solid(self):
        return("solid("+self.genCeroOneValue()+", "+self.genCeroOneValue()+", "+self.genCeroOneValue()+", "+self.genCeroPointOneToMax()+")")

    def voronoi(self):
        return("voronoi("+self.genValue()+", "+self.genValue()+", "+self.genCeroOneValue()+")")

    # END SOURCES ---


    # COLOR ---

    def brightness(self):
        return(".brightness("+self.genCeroOneValue()+")")

    def contrast(self):
        return(".contrast("+self.genCeroPointOneToMax()+")")

    def color(self):
        return(".color("+self.genCeroOneValue()+", "+self.genCeroOneValue()+", "+self.genCeroOneValue()+")")

    def colorama(self):
        return(".colorama("+self.genValue()+")")

    def invert(self):
        return(".invert("+self.genCeroOneValue()+")")

    def luma(self):
        return(".luma("+self.genCeroOneValue()+")")

    def posterize(self):
        return(".posterize("+self.genCeroOneValue()+", "+self.genCeroOneValue()+")")

    def saturate(self):
        return(".saturate("+self.genValue()+")")

    def thresh(self):
        return(".thresh("+self.genCeroOneValue()+", "+self.genCeroOneValue()+")")

    # ENDCOLOR ---


    # GEOMETRY ---

    def kaleid(self):
        return(".kaleid("+self.genValue()+")")

    def pixelate(self):
        return(".pixelate("+self.genCeroPointOneToMax()+", "+self.genCeroPointOneToMax()+")")

    def repeat(self):
        return(".repeat("+self.genValue()+", "+self.genValue()+", "+self.genValue()+", "+self.genValue()+")")

    def repeatX(self):
        return(".repeatX("+self.genValue()+", "+self.genValue()+")")

    def repeatY(self):
        return(".repeatY("+self.genValue()+", "+self.genValue()+")")

    def rotate(self):
        return(".rotate("+self.genValue()+", "+self.genValue()+")")

    def scale(self):
        return(".scale("+self.genPosOrNegValue()+", "+self.genCeroPointOneToOne()+", "+self.genCeroPointOneToOne()+")")

    def scrollX(self):
        return(".scrollX("+self.genValue()+", "+self.genValue()+")")

    def scrollY(self):
        return(".scrollY("+self.genValue()+", "+self.genValue()+")")

    # ENDGEOMETRY ---


    # MODULATORS ---

    def modulate(self):
        if(random.randint(1, 100) <= self.modulateItselfProb):
            return(".modulate(o0, " + self.genValue()+")")
        else:
            return(".modulate("+self.genSource()+", "+self.genValue()+")")

    def modulateHue(self):
        if(random.randint(1, 100) <= self.modulateItselfProb):
            return(".modulateHue(o0, " + self.genValue()+")")
        else:
            return(".modulateHue("+self.genSource()+", "+self.genValue()+")")

    def modulateKaleid(self):
        if(random.randint(1, 100) <= self.modulateItselfProb):
            return(".modulateKaleid(o0, " + self.genValue()+")")
        else:
            return(".modulateKaleid("+self.genSource()+", "+self.genValue()+")")

    def modulatePixelate(self):
        if(random.randint(1, 100) <= self.modulateItselfProb):
            return(".modulatePixelate(o0, " + self.genValue()+")")
        else:
            return(".modulatePixelate("+self.genSource()+", "+self.genValue()+")")

    def modulateRepeat(self):
        if(random.randint(1, 100) <= self.modulateItselfProb):
            return(".modulateRepeat(o0, "+self.genValue()+", "+self.genValue()+", "+self.genCeroOneValue()+", "+self.genCeroOneValue()+")")
        else:
            return(".modulateRepeat("+self.genSource()+", "+self.genValue()+", "+self.genValue()+", "+self.genCeroOneValue()+", "+self.genCeroOneValue()+")")

    def modulateRepeatX(self):
        if(random.randint(1, 100) <= self.modulateItselfProb):
            return(".modulateRepeatX(o0, "+self.genValue()+", "+self.genCeroOneValue()+")")
        else:
            return(".modulateRepeatX("+self.genSource()+", "+self.genValue()+", "+self.genCeroOneValue()+")")

    def modulateRepeatY(self):
        if(random.randint(1, 100) <= self.modulateItselfProb):
            return(".modulateRepeatY(o0, "+self.genValue()+", "+self.genCeroOneValue()+")")
        else:
            return(".modulateRepeatY("+self.genSource()+", "+self.genValue()+", "+self.genCeroOneValue()+")")

    def modulateRotate(self):
        if(random.randint(1, 100) <= self.modulateItselfProb):
            return(".modulateRotate(o0, "+self.genValue()+")")
        else:
            return(".modulateRotate("+self.genSource()+", "+self.genValue()+")")

    def modulateScale(self):
        if(random.randint(1, 100) <= self.modulateItselfProb):
            return(".modulateScale(o0, " + self.genValue()+")")
        else:
            return(".modulateScale("+self.genSource()+", "+self.genValue()+")")

    def modulateScrollX(self):
        if(random.randint(1, 100) <= self.modulateItselfProb):
            return(".modulateScrollX(o0, " +self.genCeroOneValue()+", "+self.genCeroOneValue()+")")
        else:
            return(".modulateScrollX("+self.genSource()+", "+self.genCeroOneValue()+", "+self.genCeroOneValue()+")")

    def modulateScrollY(self):
        if(random.randint(1, 100) <= self.modulateItselfProb):
            return(".modulateScrollY(o0, " +self.genCeroOneValue()+", "+self.genCeroOneValue()+")")
        else:
            return(".modulateScrollY("+self.genSource()+", "+self.genCeroOneValue()+", "+self.genCeroOneValue()+")")

    # END MODULATORS ---


    # OPERATORS ---

    def add(self):
        return(".add("+self.genSource()+", "+self.genCeroOneValue()+")")

    def blend(self):
        return(".blend("+self.genSource()+", "+self.genCeroOneValue()+")")

    def diff(self):
        return(".diff("+self.genSource()+")")

    def layer(self):
        return(".layer("+self.genSource()+")")

    def mask(self):
        return(".mask("+self.genSource()+", "+self.genValue()+", "+self.genCeroOneValue()+")")

    def mult(self):
        return(".mult("+self.genSource()+", "+self.genCeroOneValue()+")")

    # END OPERATORS ---


    def encodeText(self, fullCode):
        #fullCode= fullCode.replace(" ", '%0D')
        #fullCode= fullCode.replace("=", '%3D')
        #fullCode= fullCode.replace("\n", '%0A')
        fullCode= fullCode.replace(">", '%3E') #solves encoding interpretation problems
        fullCode = base64.b64encode(fullCode.encode())
        fullCode= str(fullCode)[2:-1]
        return fullCode

#END CODEGENERATOR CLASS


class HCGArgumentHandler(): # for better handling of codegenerator arguments in django
    fmin = 0
    fmax = 0
    amin = 0
    amax = 0
    arrow_prob = 0
    mouse_prob = 0
    modulate_itself_prob = 0
    ignore_list = []
    exclusive_source_list = []
    exclusive_function_list = []

    def get_fmin(self):
        return self.fmin
    
    def get_fmax(self):
        return self.fmax

    def get_amin(self):
        return self.amin

    def get_amax(self):
        return self.amax
    
    def get_arrow_prob(self):
        return self.arrow_prob
    
    def get_mouse_prob(self):
        return self.mouse_prob
    
    def get_modulate_itself_prob(self):
        return self.modulate_itself_prob
    
    def get_ignore_list(self):
        return self.ignore_list
    
    def get_exclusive_source_list(self):
        return self.exclusive_source_list

    def get_exclusive_function_list(self):
        return self.exclusive_function_list

    def set_fmin(self, value):
        self.fmin = value
    
    def set_fmax(self, value):
        self.fmax = value

    def set_amin(self, value):
        self.amin = value

    def set_amax(self, value):
        self.amax = value

    def set_arrow_prob(self, value):
        self.arrow_prob = value
    
    def set_mouse_prob(self, value):
        self.mouse_prob = value
    
    def set_modulate_itself_prob(self, value):
        self.modulate_itself_prob = value
    
    def set_ignore_list(self, value):
        self.ignore_list = value

    def set_exclusive_source_list(self, value):
        self.exclusive_source_list = value

    def set_exclusive_function_list(self, value):
        self.exclusive_function_list = value
