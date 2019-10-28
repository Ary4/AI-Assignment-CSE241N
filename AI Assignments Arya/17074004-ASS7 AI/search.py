"""
search.py: Search algorithms on grid.
"""


def heuristic(a, b):
    """
    Calculate the heuristic distance between two points.

    For a grid with only up/down/left/right movements, a
    good heuristic is manhattan distance.
    """

    # BEGIN HERE #
    if a is None or b is None:
        return 0
    k = abs(a[0]-b[0])
    k+= abs(a[1]-b[1])
    return k

    # END HERE #


def searchHillClimbing(graph, start, endnode):
    """
    Perform hill climbing search on the graph.

    Find the path from start to endnode.

    @graph: The graph to search on.
    @start: Start state.
    @endnode: endnode state.

    returns: A dictionary which has the information of the path taken.
             Ex. if your path contains and edge from A to B, then in
             that dictionary, d[B] = A. This means that by checking
             d[endnode], we obtain the node just before the endnode and so on.

             We have called this dictionary, the "came_from" dictionary.
    """

    # Initialise the came_from dictionary
    came_from = {}
    came_from[start] = None

    # BEGIN HERE #
    #print('chalra hai')
    came_from1 = {}
    came_from1[start] = None
    stck = [[start]]
    extnd = []
    leastN = start
    #print('1')
    while stck:
        #print('stck:{}'.format(stck))
        lvl = stck.pop()
        #print('start:{}'.format(start))
        node = leastN
        unex_lvl = list(set(lvl) - set(extnd))
        calc=1e6
        '''print('2')
        print('stck:{}'.format(stck))
        print('lvl:{}'.format(lvl))
        print('leastN:{}'.format(leastN))'''
        for a in unex_lvl:
            dist = heuristic(a, endnode) 
            if calc> dist:
                calc=dist
                leastN = a
        
        extnd.append(leastN)
        unex_lvl = list(set(lvl) - set(extnd))
        extnd=list(set(extnd))
        #print('unex_lvl:{}'.format(unex_lvl))
        unex_lvl = list(set(unex_lvl)-set(list(leastN)))
        #came_from1[leastN] = node
        if len(unex_lvl)>0:
                stck.append(unex_lvl)
        if leastN==endnode :
            #print('5')
            #print(came_from1)
            #return came_from1
            x=endnode
            while x!=start:
                #print(x)
                came_from[x]=came_from1[x]
                x = came_from1[x]
            return came_from
        else:
            tempvar1 = list(set(graph.neighboursOf(leastN))-set(extnd))
            for l in tempvar1:
                came_from1[l]=leastN
            if len(tempvar1)>0:
                stck.append(tempvar1)
        '''print('stck:{}'.format(stck))
        print()
        print()
        print()'''

            

    # END HERE #

    return came_from


def searchBestFirst(graph, start, endnode):
    """
    Perform best first search on the graph.

    Find the path from start to endnode.

    @graph: The graph to search on.
    @start: Start state.
    @endnode: endnode state.

    returns: A dictionary which has the information of the path taken.
             Ex. if your path contains and edge from A to B, then in
             that dictionary, d[B] = A. This means that by checking
             d[endnode], we obtain the node just before the endnode and so on.

             We have called this dictionary, the "came_from" dictionary.
    """


    # Initialise the came_from dictionary
    came_from = {}
    came_from[start] = None


    # BEGIN HERE #
    '''p=[1,2,3,4]
    c=10
    for idx, a in enumerate(p):
        if c>a:
            c=a
            i=idx
    p.pop(i)
    print(p)'''
    import heapq
    if start==endnode:
        return came_from
    vist={}
    parent={}
    pq=[]
    pq.append((heuristic(start,endnode),start))
    vist[start]=1
    heapq.heapify(pq)
    currnode=None
    while(len(pq)):
        next_node = heapq.heappop(pq)[1]
        if next_node == endnode:
            break
        currnode=next_node
        c_neighbors = graph.neighboursOf(currnode)
        for neighbor in c_neighbors:
            if not vist.get(neighbor,0):
                parent[neighbor]=currnode
                vist[neighbor]=1
                heapq.heappush(pq,(heuristic(neighbor,endnode),neighbor))
    if endnode not in parent.keys():
        return came_from
    tempvar=endnode
    while(1):
        came_from[tempvar]=parent[tempvar]
        tempvar=parent[tempvar]
        if(tempvar==start):
            return came_from


    # END HERE #

    return came_from



def searchBeam(graph, start, endnode, beam_length=3):
    """
    Perform beam search on the graph.

    Find the path from start to endnode.

    @graph: The graph to search on.
    @start: Start state.
    @endnode: endnode state.

    returns: A dictionary which has the information of the path taken.
             Ex. if your path contains and edge from A to B, then in
             that dictionary, d[B] = A. This means that by checking
             d[endnode], we obtain the node just before the endnode and so on.

             We have called this dictionary, the "came_from" dictionary.
    """

    # Initialise the came_from dictionary
    came_from = {}
    came_from[start] = None

    # BEGIN HERE #
    #print('start:{}'.format(start))

    '''came_from1 = {}
    came_from1[start] = None
    queue = [[start]]
    extnd = []
    leastN = start
    while queue:
        #print('queue:{}'.format(queue))
        lvl = queue.pop()
        #print('lvl:{}'.format(lvl))
        node=leastN
        #print('naya')
        candidate_nodes=[]
        extnd+=lvl
        extnd = list(set(extnd))
        unex_lvl=lvl
        #print('unex_lvl:{}'.format(unex_lvl))
        #unex_lvl = list(set(lvl) - set(extnd))
        for ek in unex_lvl:
            #print('ek:{}'.format(ek))
            for child in list(set(graph.neighboursOf(ek))-set(extnd)):
                #print('child:{}'.format(child))
                candidate_nodes.append([child, ek, heuristic(child, endnode)])
                #print('1')
        selected=[]
        flag = False
        #print('candidate_nodes:{}'.format(candidate_nodes))
        for idx in range(beam_length):
            if len(list(candidate_nodes))==0:
                flag = True
                break
            calc=1e6
            for cand in candidate_nodes:
                dist = cand[2] 
                if calc> dist:
                    calc=dist
                    leastN = cand
            # extnd.append(leastN)
            # extnd = list(set(extnd))
            selected.append(leastN[0])
            
            # print('idx:{}'.format(idx))
            # print('candidate_nodes:{}'.format(candidate_nodes))
            # print('leastN:{}'.format(leastN))
            came_from1[leastN[0]] = leastN[1]
            for a in candidate_nodes:
                if a[0]==leastN[0]:
                    del a
            
            #print('extnd:{}'.format(extnd))
            #print('leastN:{}'.format(leastN))
            #print('selected:{}'.format(selected))
            #unex_lvl = list(set(unex_lvl)-set(list(leastN)))
            
            if leastN == endnode :
                #print('5')
                x=endnode
                #return came_from1
                while x!=start:
                    came_from[x]=came_from1[x]
                    x = came_from1[x]
                    #print('hmm')
                return came_from
            if len(candidate_nodes)==0:
                flag = True
                break

        selected = list(set(selected))
        if len(selected):
            queue.append(selected)'''
        #print('queue:{}'.format(queue))

    '''import heapq
    x=graph.height
    y=graph.width
    extnd=[[0 for i in range(x)] for j in range(y)]
    lvl=[start]
    extnd[start[0]][start[1]]=1
    paren={}
    while lvl:
        dist=[]
        for i in lvl:
            neigh = list(set(graph.neighboursOf(i)))
            for j in neigh:
                if extnd[j[0]][j[1]]!=0:
                    extnd[j[0]][j[1]]=1
                    paren[j]=i
                    dist.append((heuristic(j,endnode),j))
        dist.sort()
        lvl=[]
        m= max(len(dist),beam_length)
        hmm= len(dist)+beam_length-m
        for i in range(hmm):
            s.append(a[i][1])
    if endnode in paren:
        p=endnode
        while p!=start:
            came_from[p]=paren[p]
            p=paren[p]'''

    l = []
    import copy
    g = copy.deepcopy(graph)
    l.append((heuristic(start,endnode),start))
    flag = True
    # cf = dict()
    while flag and len(l)>0:
        c = []

        for i in l:
            g.oob.append((i[1],i[1]))
            if(i[1]==endnode):
                break
            z = g.neighboursOf(i[1])
            for _ in z:
                came_from[_]=i[1]
                # g.oob.append((_,_))
                # if _ == endnode:
                #     flag = False
                #     break
            # g.oob.append((i[1],i[1]))
            z = map(lambda x : (heuristic(x,endnode),x),z)
            c.extend(z)
        c.sort(key=lambda x : x[0])
        # if len(c)>beam_length:
        l = c[:beam_length]
        # for i in l:
        #     g.oob.append((i[1],i[1]))
        # else:
        #     l = c
    cf = dict()
    # print(type(endnode))
    # input()
    if endnode not in came_from.keys():
        return cf
    if start ==endnode:
        return cf
    cf[start]=None
    while endnode is not start:
        cf[endnode] = came_from[endnode]
        endnode = came_from[endnode]
    return cf     

    # END HERE #

    return came_from


def searchAStar(graph, start, endnode):
    """
    Perform A* search on the graph.

    Find the path from start to endnode.

    @graph: The graph to search on.
    @start: Start state.
    @endnode: endnode state.

    returns: A dictionary which has the information of the path taken.
             Ex. if your path contains and edge from A to B, then in
             that dictionary, d[B] = A. This means that by checking
             d[endnode], we obtain the node just before the endnode and so on.

             We have called this dictionary, the "came_from" dictionary.
    """

    # Initialise the came_from dictionary
    came_from = {}
    came_from[start] = None

    # BEGIN HERE #
    '''came_from1 = {}
    came_from1[start] = None
    prioqueue=[]
    extnd = []
    leastN = [start, None, heuristic(start, endnode)]
    prioqueue = [leastN]
    #print('prioqueue:{}'.format(prioqueue))
    while prioqueue:
        calc=1e6
        for idx,a in enumerate(prioqueue): 
            if calc> a[2]:
                calc=a[2]
                leastN = a
                i=idx
        #print('1')
        #print('prioqueue:{}'.format(prioqueue))
        extnd.append(leastN[0])
        prioqueue.pop(i)
        #print('extnd:{}'.format(extnd))

        for child in graph.neighboursOf(leastN[0]):
            if child==endnode:
                x=endnode
                came_from1[endnode]=leastN[0]
                #return came_from1
                while x!=start:
                    #print(x)
                    came_from[x]=came_from1[x]
                    x = came_from1[x]
                return came_from
            if child not in extnd:
                prioqueue.append([child, leastN[0], (leastN[2]-heuristic(leastN[1],endnode)+1+heuristic(child, endnode))])
                came_from1[child] = leastN[0]'''
    import heapq
    if start==endnode:
        return came_from
    vist={}
    paren={}
    prioqueue=[]
    d=0
    prioqueue.append((heuristic(start,endnode),start))
    vist[start]=1
    heapq.heapify(prioqueue)
    currnode=None
    while(len(prioqueue)):
        h,next_node = heapq.heappop(prioqueue)
        if next_node == endnode:
            break
        currnode=next_node
        c_neighbors = graph.neighboursOf(currnode)
        for neighbor in c_neighbors:
            if neighbor == endnode:
                paren[neighbor]=currnode
            if not vist.get(neighbor,0):
                vist[neighbor]=1
                paren[neighbor]=currnode
                d=h-heuristic(currnode,endnode)+1
                heapq.heappush(prioqueue,(d+heuristic(neighbor,endnode),neighbor))
    if endnode not in paren.keys():
        return came_from
    tempvar=endnode
    while(1):
        came_from[tempvar]=paren[tempvar]
        tempvar=paren[tempvar]
        if(tempvar==start):
            return came_from

                
            
        

    # END HERE #

    return came_from

