import copy 
class Rule:
	def __init__(self,head,body=None):
		self.head=head
		self.body=body
	
	def replaceVARs(s):
		newOperands =[]
		for i in range(self.operandCount()):
			newOperands.append(self.getOperand(i).replaceVARs(s))

		copy_temp=copy.deepcopy(self)
		copy_temp.setOperands(newOperands)
		return copy_temp
	def getHead(self):
		return self.head

	def getBody(self):
		return self.body


	def toString(self):
		if (self.body == null):
			return self.head.toString()
		return self.head + " :- " + self.body



class VAR():
	
	nextId = 1
	
	def __init__(self,printName=None):
		
			self.printName= printName
		
			self.myid = VAR.nextId
			VAR.nextId+=1
	def toString(self):
		if self.printName!= None: return self.printName+"_"+str(self.myid)
		return "VAR_" + str(self.myid)
	
	def unify(self,p,s):
		if (self == p): return s
		if (s.isBound(self)): return s.getBinding(self).unify(p, s)
		sNew = SUBSTITUTSET(s.bindings)  # debug here?????
		sNew.add(self, p)
		return sNew

	def replaceVARs(self,s):
		if ( s.isBound(self)): return s.getBinding(self).replaceVARs(s)
		else:
			return self

	def standardizevariablesApart(self,newVars):
		if self in newVars.keys(): newvar=newVars[self]
		else:
			newvar=VAR(self)
			newVars[self]=newvar
		return newvar

class CONST():
	
	nextId = 0  
	
	def __init__(self,printName=None):
		
		self.printName= printName
		
		self.myid = CONST.nextId
		CONST.nextId+=1

	def toString(self):
		if self.printName!= None: return self.printName
		return "CONST_" + str(self.myid)
	def unify(self,exp,s):
		if (self == exp): return SUBSTITUTSET(s.bindings)  # correction??
		if isinstance(exp,VAR): return exp.unify(self, s)
		return None
	def replaceVARs(self,s): return self

	def standardizevariablesApart(self,newvars): return self



class SIMPLSENT():
	def __init__(self,*args):
		self.terms=[]
		for i in args: self.terms.append(i)
		

	def __repr__(self):
		return str(self.__dict__)
		
	def toString(self):
		s = None
		for p in range(len(self.terms)):
			if s == None:
				# debug print("self.terms[p]",self.terms[p])
				s = self.terms[p].toString()
			else:
				s += " " + self.terms[p].toString()
		if s == None: return "None"
		return "(" + s + ")"
	def length(self):
		return len(self.terms)
	def getTerm(self,index):
		return self.terms[index]

	def unify(self,p,s):
		if isinstance(p,SIMPLSENT):
			# s2 = SIMPLSENT(p) #type-casting p just like int(p)
			s2=p
			if (self.length() != s2.length()):
				return None
			# SUBSTITUTSET sNew = new SUBSTITUTSET(s)
			sNew =SUBSTITUTSET(s.bindings)
			for i in range(self.length()):
				sNew = (self.getTerm(i)).unify(s2.getTerm(i),sNew)
				if (sNew == None):
					return None
			return sNew
		if isinstance(p,VAR):
			return p.unify(self, s)
		return None
	def replaceVARs(self,s):
		# [] newTerms = new [terms.length];
		newTerms = [None]*self.length()
		for i in range(self.length()):
			newTerms[i] =((self.getTerm(i)).replaceVARs(s))
		
		return SIMPLSENT(*newTerms)

	def standardizevariablesApart(self,newvars):
		newTerms=[None]*SIMPLSENT.length(self)
		for i in range(SIMPLSENT.length(self)): newTerms[i]=self.terms[i].standardizevariablesApart(newVars)
		return SIMPLSENT(*newterms)

	def getSolver(self,rules,solution): return SIMPLSENTSolutionNode(None,rules,solution)


class SUBSTITUTSET:
	# private HashMap<VAR, > bindings =new HashMap<VAR, >()
	
	def __init__(self,s={}):
			self.bindings=s

	def clear(self):
		self.bindings={}
	def add(self,v,exp):
		self.bindings[v]=exp
	def getBinding(self,v):
		return (self.bindings[v])
	def isBound(self,v):
		return (v in self.bindings.keys())
	def toString(self):
		return "Bindings:[" + self.bindings + "]"
	
class UnifyTester:
	def main(args):
		
		friend =  CONST("friend")
		bill =  CONST("bill")
		george =  CONST("george")
		kate =  CONST("kate")
		merry =  CONST("merry")
		X =  VAR("X")
		Y =  VAR("Y")
		# Vector<> expressions =
		#  Vector<>()
		expressions=[]
		expressions.append(SIMPLSENT(friend,
		bill, george))
		expressions.append(SIMPLSENT(friend,
		bill, kate))
		expressions.append(SIMPLSENT(friend,
		bill, merry))
		expressions.append(SIMPLSENT(friend,
		george, bill))
		expressions.append(SIMPLSENT(friend,
		george, kate))
		expressions.append(SIMPLSENT(friend,
		kate, merry))
		# //Test 1
		goal =  SIMPLSENT(friend, X,
		Y)
	
		# Iterator iter = expressions.iterator()
		
		print("Goal = " + goal.toString())
		for iter in expressions:
			next = (iter)
			x=  SUBSTITUTSET({})
			s = next.unify(goal,x)
			if(s != None):
				print(goal.replaceVARs(s).toString())
			else:
				print("False")

	# //Test 2
		goal =  SIMPLSENT(friend, bill, Y)
		# iter = expressions.iterator()
		print("Goal = " + goal.toString())


		for iter in expressions:
			next = (iter)
			x=  SUBSTITUTSET({})
			s = next.unify(goal,x)

			if (s != None):
				print(goal.replaceVARs(s).toString())
				
			else:
				print("False")
			



class AbstractOperator:
	# operands=[]
	def __init__(self,*operands):  #unpack lists using * operator before sending here
		operandArray = operands
		self.operands=[]
		for i in operands:
			self.operands.append(i)

	def setOperands(self,operands):
		self.operands=operands

	def operandCount(self):
		return len(self.operands)

	def getOperand(self,i):
		return self.operands[i]

	def getFirstOperand(self):
		return self.operands[0]

	def getOperatorTail(self):
		tail=self.operands[1:]
		tailOperator = copy.deepcopy(self)
		tailOperator.setOperands(tail)
		return tailOperator

	def isEmpty(self):
		return (not self.operands)

	def replaceVARs(self, s):
		newOperands =[]
		for i in range(self.operandCount()):
			newOperands.append(self.getOperand(i).replaceVARs(s))

		copy_temp=copy.deepcopy(self)
		copy_temp.setOperands(newOperands)
		return copy_temp

	def standardizevariablesApart(wVars):
		newOperands=[]
		for i in range(self.operandCount()):
			newOperands.append(self.getOperand(i).standardizevariablesApart({}))
		copy_temp=copy.deepcopy(self)
		copy_temp.setOperands(newOperands)
		return copy_temp


class And(AbstractOperator):
	def __init__(self,* operands):  # unpack before sending list
		super(And,self).__init__(operands)

	def toString(self):
		result="(AND "
		for i in range(self.operandCount()):
			result = result + self.getOperand(i).toString()
			return result

	

class RuleSet:
	def __init__(self,*rules):
		self.rules=rules

	def getRuleStandardizedApart(self, i):
		rule=self.rules[i].standardizevariablesApart({})
		return rule
	def getRule(self,i):
		return self.rules[i]

	def getRuleCount(self):
		return len(self.rules)

	
class AbstractSolutionNode:
	def __init__(self,goal,rules,parentSolution):
		self.goal=goal
		self.rules=rules
		self.parentSolution=parentSolution
		self.ruleNumber = 0
		self.currentRule =None

	
	def getRuleSet(self):
		return self.rules
	def getcurrentRule(self):
		return self.currentRule
	def reset(self,newParentSolution):
		self.parentSolution = newParentSolution
		self.ruleNumber = 0

	def nextRule(self):
		if(self.hasNextRule()):
			self.currentRule =self.rules.getRuleStandardizedApart(self.ruleNumber)
			self.ruleNumber+=1
		else:
			self.currentRule =None
		return self.currentRule

	def hasNextRule(self):
		return self.ruleNumber < self.rules.getRuleCount()
	def getParentSolution(self):
		return self.parentSolution
	def getGoal():
		return self.goal

class SIMPLSENTSolutionNode(AbstractSolutionNode):
	def __init__(self,goal,rules,parentSolution):
		super(SIMPLSENTSolutionNode,self).__init__(goal,rules,parentSolution)
		self.child = None
	def getChild(self):
		return self.child
	def nextSolution(self):
		if(self.child != None):
			solution = self.child.nextSolution()
			if (solution != None):
				return solution
		self.child = None
		while(self.hasNextRule() == True):
			rule = self.nextRule()
			head = rule.getHead()
			solution = self.goal.unify(head,self.getParentSolution())
			if(solution != None):
				tail = rule.getBody()
				if(tail == None):
					return solution
				child = tail.getSolver(self.getRuleSet(),solution)
				childSolution =child.nextSolution()
				if(childSolution != None):
					return childSolution
		return None
	
	
class AndSolutionNode(AbstractSolutionNode):
	def __init__(self,goal,rules,parentSolution):
		super(AndSolutionNode,self).__init__(goal,rules,parentsolution)
		self.headSolutionNode =goal.getFirstOperand().getSolver(rules, parentSolution)
		self.operatorTail = goal.getOperatorTail()
		self.tailSolutionNode =None

	def nextSolution(self):
		solution=None
		if(tailSolutionNode != None):
			solution =self.tailSolutionNode.nextSolution()
			if(solution != None):
				return solution
		solution =self.headSolutionNode.nextSolution()
		while (solution)!= None :
			if(self.operatorTail.isEmpty()):
				return solution
			else:
				self.tailSolutionNode = self.operatorTail.getSolver(self.getRuleSet(), solution)
				tailSolution =self.tailSolutionNode.nextSolution()
				if(tailSolution != None) :
					return tailSolution
			solution =self.headSolutionNode.nextSolution()
		return None

	def getHeadSolutionNode(self):
		return self.headSolutionNode
	def getTailSolutionNode(self):
		return self.tailSolutionNode

class Tester:
	def main(args):
		parent = CONST("parent")
		bill =  CONST("Bill")
		audrey =  CONST("Audrey")
		maria =  CONST("Maria")
		tony =  CONST("Tony")
		charles =  CONST("Charles")
		ancestor =  CONST("ancestor")
		X =  VAR("X")
		Y =  VAR("Y")
		Z =  VAR("Z")
		rules = []
		rules.append( Rule( SIMPLSENT(parent,bill, audrey)))
		rules.append( Rule( SIMPLSENT(parent,maria, bill)))
		rules.append( Rule( SIMPLSENT(parent,tony,maria)))
		rules.append( Rule( SIMPLSENT(parent,charles,tony)))
		rules.append( Rule( SIMPLSENT(ancestor,X,Y), And( SIMPLSENT(parent,X, Y))))
		rules.append( Rule( SIMPLSENT(ancestor,X,Y),And( SIMPLSENT(parent,X, Z),SIMPLSENT(ancestor, Z, Y))))
		rules=RuleSet(*rules)
		goal =SIMPLSENT(ancestor,charles, Y)
		root =goal.getSolver(rules,SUBSTITUTSET({}))
		solution=None
		print(("Goal = " + goal.toString()))
		print(("Solutions:"))
		solution = root.nextSolution()
		while solution!=None:
			print((" " + goal.replaceVARs(solution)))
			solution = root.nextSolution()




obj=UnifyTester()
obj.main()
