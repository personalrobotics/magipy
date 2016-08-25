import pylab, threading
import matplotlib.pyplot as plt

def get_action_name(action):
    action_name = action.get_name()
    action_name = action_name.replace("_", "-")
    if action.checkpoint:
        action_name += " (Checkpoint)"
    return action_name

class ActionResults(object):
    # Unknown - result hasn't been reported yet
    UNKNOWN = -1
    
    # Planning for the action failed
    FAILURE = 0

    # Planning for the action succeeded and the solution
    #  is deterministic
    DETERMINISTIC_SUCCESS = 1
    
    # Planning for the action succeeded but the solution
    #  is non-deterministic
    NONDETERMINISTIC_SUCCESS = 2

def get_color(result):
    """
    An action result
    """
    # Store colors for success, failure and unknown
    # colors taken from the palettable package
    # https://jiffyclub.github.io/palettable/
    # colorbrewer = palettable.colorbrewer
    color_map = {
        ActionResults.FAILURE: [215, 48, 39], # red
        ActionResults.DETERMINISTIC_SUCCESS: [49, 130, 189], # blue
        ActionResults.NONDETERMINISTIC_SUCCESS: [26, 152, 80] # green
    }

    color = color_map.get(result, [240, 240, 240]) # unknown=grey
    return [float(c)/255. for c in color]

class MonitorUpdateRequest(object):
    # Used to indicate to the drawing thread that
    #  data has been updated
    pass

class MonitorStopRequest(object):
    # Used to indicate the drawing thread
    # should close the window and exit
    pass

class ActionMonitor(object):
    
    def __init__(self):

        # Create a lock for thread safety
        self.lock = threading.Lock()

        # Create an internal queue to be used by the internal
        # draw and upate thread
        from Queue import Queue
        self.update_request_queue = Queue()

        # Setup internal data structures
        self.reset()

        # Spin off drawing thread
        self.drawing_thread = threading.Thread(target=self._draw)
        self.drawing_thread.start()

    def stop(self):
        self.update_request_queue.put(MonitorStopRequest())
        self.drawing_thread.join()
        
    def _draw(self):
        self.ax_graph = pylab.subplot(121)
        self.ax_bar = pylab.subplot(122)
        plt.show(block=False)
        plt.rcParams.update({
            'font.family':'serif',
            'font.serif':'Computer Modern Roman',
            'font.size': 12,
            'legend.fontsize': 12,
            'legend.labelspacing': 0,
            'text.usetex': True,
            'savefig.dpi': 300
        })

        # Set the window title
        plt.gcf().canvas.set_window_title('HGPC Monitor')

        keep_running = True
        while keep_running:
            self._redraw()
            r = self.update_request_queue.get()
            keep_running = not isinstance(r, MonitorStopRequest)
            
        plt.close()

    def reset(self):
        self.lock.acquire()
        try:
            self.node_metadata = {}
            self.current_planning = None # planning
            self.current_post_processing = None # post processing
            self.current_executing = None # executing
            self.planned_actions = [] # Keep track of the actions in the order they are planned
            self.active_actions = []
                        
            # Initialize a graph for the action ordering
            import networkx as nx
            self.G = nx.DiGraph(format='svg')
        finally:
            self.lock.release()

    def set_graph(self, G, node_map):
        """
        @param G A networkx graph representing the relationship between actions
        @param node_map A mapping from node name in G to action
        """
        self.lock.acquire()
        try:
            self.G = G
            for n in self.G.nodes():
                if n not in self.node_metadata:
                    self.node_metadata[n] = {'color': get_color(ActionResults.UNKNOWN),
                                             'results': [] }
            self.node_map = node_map
            self.action_map = {v:k for k,v in node_map.iteritems() }
            self.update_request_queue.put(MonitorUpdateRequest())
        finally:
            self.lock.release()

    def _draw_dot_plots(self):

        # Clear the existing axis
        self.ax_bar.cla()

        import numpy
        categories = self.planned_actions
        if len(categories) == 0:
            return 

        # Build a set of nodes with a custom layout
        import networkx as nx
        G = nx.Graph()
        node_pos = {}
        label_map = {}
        label_pos = {}
        color_map = {}
        node_spacing=0.2

        for idx, category in enumerate(categories):
            
            # Add nodes for every result
            results = self.node_metadata[category]['results']
            for ridx, r in enumerate(results):
                node_label = '%d_%d' % (idx, ridx)
                G.add_node(node_label)
                color_map[node_label] = get_color(r)
                node_pos[node_label] = (0.2*ridx, idx)
            if len(results):
                node_label = '%d_0' % (idx)
                action = self.node_map[category]
                label_map[node_label] = get_action_name(action)
                label_pos[node_label] = (0.2*len(results), idx) 

        nodesize=50
        fontsize=10

        max_x = max([0.0] + [v[0] for v in node_pos.values()])
        label_pos = {k: (max_x+node_spacing, v[1]) for k,v in label_pos.iteritems()}

        color_list = [color_map[a] for a in G.nodes()]
        nx.draw_networkx_nodes(G,
                               node_pos,
                               ax=self.ax_bar,
                               nodelist=G.nodes(),
                               node_color=color_list,
                               node_size=nodesize)
        
        labels=nx.draw_networkx_labels(G, label_pos, label_map, 
                                ax=self.ax_bar,
                                font_size=fontsize,
                                font_family='serif', 
                                horizontalalignment='left')
        
        
        self.ax_bar.get_xaxis().set_visible(False)
        self.ax_bar.get_yaxis().set_visible(False)
        self.ax_bar.set_title('Planning Results by Action')

        max_y = max([0.0] + [v[1] for v in node_pos.values()])
        max_x = max([0.0] + [v[0] for v in node_pos.values()])
        self.ax_bar.set_ylim((-1., max_y+1))
        self.ax_bar.set_xlim((-0.2, max_x+1.5))
        
    def _draw_bargraphs(self):

        def compute_success(results):
            return sum([1 if r == ActionResults.DETERMINISTIC_SUCCESS or \
                        r == ActionResults.NONDETERMINISTIC_SUCCESS \
                        else 0 for r in results])
        def compute_failures(results):
            return len(results) - compute_success(results)

        import numpy
        categories = self.planned_actions
        if len(categories) == 0:
            return 

        bar_height = 1.0
        category_positions = numpy.arange(bar_height*1.5, 
                                          1.5*bar_height*len(categories) + 0.1, 
                                          1.5*bar_height)

        
        failure_counts = [compute_failures(self.node_metadata[c]['results']) for c in categories ]
        success_counts = [compute_success(self.node_metadata[c]['results']) for c in categories]

        plt.hold(False)
        self.ax_bar.barh(category_positions, failure_counts,
                     align='center',
                     height=bar_height,
                     color=get_color(ActionResults.FAILURE),
                     lw=0)
        plt.hold(True)
        self.ax_bar.barh(category_positions, success_counts,
                     align='center',
                     height=bar_height,
                     color=get_color(ActionResults.NONDETERMINISTIC_SUCCESS),
                     lw=0)
        self.ax_bar.set_yticks([])

        for idx, category in enumerate(categories):
            action = self.node_map[category]
            self.ax_bar.text(0.1, category_positions[idx], r'%s' % get_action_name(action), va='center', ha='left')
            
        max_xval = max([len(c['results']) for c in self.node_metadata.itervalues()])
        delta = 1 if max_xval < 10 else 5
        self.ax_bar.set_xticks(range(0, max_xval, delta) + [max_xval])
        self.ax_bar.set_xlabel('Planning Counts')

        self.ax_bar.set_xlim((0, max(failure_counts) + 0.5))
        self.ax_bar.set_ylim([category_positions[0] - bar_height, category_positions[-1] + bar_height])

        # Simplify axis
        self.ax_bar.set_frame_on(False)
        xmin, xmax = self.ax_bar.get_xaxis().get_view_interval()
        ymin, ymax = self.ax_bar.get_yaxis().get_view_interval()
        self.ax_bar.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax), 
                                      color='black', linewidth=1, 
                                      zorder=100, clip_on=False))
        self.ax_bar.add_artist(plt.Line2D((xmax, xmax), (ymin, ymax), 
                                   color='black', linewidth=1, 
                                   zorder=100, clip_on=False))
        self.ax_bar.add_artist(plt.Line2D((xmin, xmax), (ymin, ymin), 
                                      color='black', linewidth=1, 
                                      zorder=100, clip_on=False))
        self.ax_bar.add_artist(plt.Line2D((xmin, xmax), (ymax, ymax), 
                                      color='black', linewidth=1, 
                                      zorder=100, clip_on=False))
       
        self.ax_bar.get_yaxis().tick_left()
        self.ax_bar.get_xaxis().tick_bottom()

    def _compute_layout(self, node, branch_id, pos_layout, label_layout, depth=0):
        if node not in pos_layout:
            pos_layout[node] = (branch_id, depth)
            label_layout[node] = (branch_id, depth)
        children = self.G.successors(node)
        for idx, s in enumerate(children):
            self._compute_layout(s, branch_id+2.*idx/(1.0*depth+5.0),
                                 pos_layout, label_layout, depth+1)

    def _draw_graph(self):
        """
        Draw the tree as a graph
        """

        # Clear the existing axis
        self.ax_graph.cla()

        # Grab all the nodes and setup a unique layout
        nodelist = self.G.nodes()

        if len(nodelist) == 0:
            # nothing to draw yet
            return

        small_nodesize=2000/len(nodelist)
        large_nodesize=max(200, 2.*small_nodesize)
        fontsize=10

        color_list = [self.node_metadata[a]['color'] for a in nodelist]
        size_list = [large_nodesize if a == self.current_planning or \
                     a == self.current_post_processing or \
                     a == self.current_executing
                     else small_nodesize for a in nodelist]

        label_dict = { a:""  for a in nodelist}
        if self.current_planning is not None:
            action = self.node_map[self.current_planning]
            label_dict[self.current_planning] = '%s' % get_action_name(action)

        if self.current_post_processing is not None and self.current_post_processing in label_dict:
            action = self.node_map[self.current_post_processing]
            label_dict[self.current_post_processing] = '%s (post-processing)' % get_action_name(action)

        if self.current_executing is not None and self.current_executing in label_dict:
            action = self.node_map[self.current_executing]
            label_dict[self.current_executing] = '%s (executing)' % get_action_name(action)

        fontsize_list = [fontsize*2. if a == self.current_planning else fontsize for a in nodelist]

        import networkx as nx
        root_nodes = [n for n in self.G.nodes() if self.G.in_degree(n) == 0 ]
        
        depth = 0
        node_pos = dict()
        label_pos = dict()
        for idx, n in enumerate(root_nodes):
            self._compute_layout(n, 0.2*idx, node_pos, label_pos)
        
        max_out_degree = max([max(self.G.in_degree(n), self.G.out_degree(n)) for n in self.G.nodes()])
        
        for k,v in label_pos.iteritems():
            label_pos[k] = (0.2*max_out_degree+0.1, v[1])

        nx.draw_networkx_nodes(self.G,
                               node_pos,
                               ax=self.ax_graph,
                               nodelist=self.G.nodes(),
                               node_color=color_list,
                               node_size=size_list)
        nx.draw_networkx_edges(self.G, node_pos, ax=self.ax_graph, edgelist=self.G.edges(), 
                               edge_color='k', width=1., alpha=0.5)
        nx.draw_networkx_labels(self.G, label_pos, label_dict, ax=self.ax_graph, font_size=fontsize,
                                font_family='serif', horizontalalignment='left')

        self.ax_graph.get_xaxis().set_visible(False)
        self.ax_graph.get_yaxis().set_visible(False)
        self.ax_graph.set_title('Planning Progress')

        max_y = max([v[1] for v in node_pos.values()])
        self.ax_graph.set_ylim((-1., max_y+1))
        self.ax_graph.set_xlim((-0.2, max_out_degree+0.5))

    def _redraw(self):
        """
        Redraw the graphs. Should be called
        when any metadata is updated
        """
        self._draw_graph()
        self._draw_dot_plots()
        pylab.draw()
     
    def set_planning_action(self, action):
        """
        Mark the action that is currently being planned
        by the system
        """
        self.lock.acquire()
        try:
            # Get the node id in the graph for this action
            node_id = self.action_map.get(action, None)
            
            # Mark this action as currently being planned
            self.current_planning = node_id
            
            if node_id is not None and node_id not in self.planned_actions:
                self.planned_actions.append(node_id)
            self.update_request_queue.put(MonitorUpdateRequest())

        finally:
            self.lock.release()

    def set_post_processing_action(self, action):
        """
        Mark the action that is currently being post-processed
        by the system
        """
        self.lock.acquire()
        try:
            # Get the node id in the graph for this action
            node_id = self.action_map.get(action, None)
            
            # Mark this action as currently being post-processed
            self.current_post_processing = node_id
            self.update_request_queue.put(MonitorUpdateRequest())
        finally:
            self.lock.release()


    def set_executing_action(self, action, redraw=True):
        """
        Mark the action that is currently being executed
        by the system
        """
        self.lock.acquire()
        try:
            # Get the node id in the graph for this action
            node_id = self.action_map.get(action, None)
           
            # Mark this action as currently being executed
            self.current_executing = node_id
            self.update_request_queue.put(MonitorUpdateRequest())
        finally:
            self.lock.release()

    def update(self, action, success, deterministic):
        """
        Update the counts on success or failed planning for the action
        @param action The Action to update
        @param success True if a plan was successfully generated for the Action
        @param deterministic True if the method to solve the action is deterministic
        """
        self.lock.acquire()
        try:
            node_id = self.action_map[action]


            if node_id not in self.node_metadata:
                self.node_metadata[node_id] = {'color': get_color(ActionResults.UNKNOWN),
                                               'results': [] }
        
            if success:
                if deterministic:
                    result = ActionResults.DETERMINISTIC_SUCCESS
                else:
                    result = ActionResults.NONDETERMINISTIC_SUCCESS
            else:
                result = ActionResults.FAILURE
            
            self.node_metadata[node_id]['results'] += [ result ]
            self.node_metadata[node_id]['color'] = get_color(result)
            self.update_request_queue.put(MonitorUpdateRequest())
        finally:
            self.lock.release()
