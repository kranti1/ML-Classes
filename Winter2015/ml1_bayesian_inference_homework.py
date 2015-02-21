################################################################################
#
#  MLEARN 210 : homework #4  ::  Bayesian Inference
#
################################################################################


################################################################################
################################################################################


################################################################################
#
#  GLOBAL DATA
#


#
#  The PREMISE is that a person may have bought one or more of the following gadgets:
#      mobile Phone, Walkman, and/or Radio  (P, W, R).
#
#  He has limited funds, so buying one diminished his likelihood of having bought another --
#  but not in a causal loop, as Bayes Net must be acyclic.
#  He passed the Phone, Walkman, and Radio stores in that order; and causal links work only forward in time.
#
#  However buying any of them increases his chances of buying Batteries (B).
#  He may then buy a Charger (C) ; but that's less likely if he has batteries.
#
#  This defines the Bayes Net.  SEE THE ACCOMPANYING IMAGE.
#

#
#  g_COND_PROBS  - map of conditional probabilities
#
#  This defines the BAYES NET in terms of the INITIAL state of the conditional probability tables.
#
#  Unlike in drawn probability tables, which have "T" and "F" columns for true/false probabilities;
#  in the code, we only store the probability of a variable being TRUE, and the probability of the negation (false) is implied.
#
#  The encoded probabilities tables are:
#      ----------------------------------------
#          P(PHONE) = 0.1
#
#          P(WALKMAN | PHONE) = 0.05
#          P(WALKMAN | !PHONE) = 0.1
#
#          P(RADIO | (PHONE and WALKMAN) ) = 0.2
#          P(RADIO | (PHONE and !WALKMAN) ) = 0.4
#          P(RADIO | (!PHONE and WALKMAN) ) = 0.3
#          P(RADIO | (!PHONE and !WALKMAN) ) = 0.6
#
#          P(BATTERIES | (PHONE and RADIO and WALKMAN) ) = 0.8
#          P(BATTERIES | (PHONE and RADIO and !WALKMAN) ) = 0.7
#          P(BATTERIES | (PHONE and !RADIO and WALKMAN) ) = 0.5
#          P(BATTERIES | (PHONE and !RADIO and !WALKMAN) ) = 0.4
#          P(BATTERIES | (!PHONE and RADIO and WALKMAN) ) = 0.6
#          P(BATTERIES | (!PHONE and RADIO and !WALKMAN) ) = 0.3
#          P(BATTERIES | (!PHONE and !RADIO and WALKMAN) ) = 0.2
#          P(BATTERIES | (!PHONE and !RADIO and !WALKMAN) ) = 0.0
#
#          P(CHARGER | BATTERIES) = 0.4
#          P(CHARGER | !BATTERIES) = 0.9
#      ----------------------------------------
#
# (Note that, for consistency, the concocted key strings are in ALPHABETICAL order wrt the root variable names;
#  so the code can generate the key DETERMINISTICALLY).
#
g_COND_PROBS = {

    'PHONE' : { "":0.1 },      # (this is the prior probability for the root variable PHONE; it depends on nothing)

    'WALKMAN' : { "PHONE":0.05, "!PHONE":0.1 },

    'RADIO' : { "PHONE,WALKMAN":0.2,  "PHONE,!WALKMAN":0.4,  "!PHONE,WALKMAN":0.3,  "!PHONE,!WALKMAN":0.6  },

    'BATTERIES' : {                             
                              "PHONE,RADIO,WALKMAN":0.8,  "PHONE,RADIO,!WALKMAN":0.7,  "PHONE,!RADIO,WALKMAN":0.5,  "PHONE,!RADIO,!WALKMAN":0.4,  
                              "!PHONE,RADIO,WALKMAN":0.6,  "!PHONE,RADIO,!WALKMAN":0.3,  "!PHONE,!RADIO,WALKMAN":0.2,  "!PHONE,!RADIO,!WALKMAN":0.0
                         },

    'CHARGER' : { "BATTERIES":0.4, "!BATTERIES":0.9 },
  }


#
#  These are just lookup tables that collect all variables, and map one-char abbreviations to the actual variable names.
#
g_ALL_VARIABLES = [ "PHONE", "WALKMAN", "RADIO", "BATTERIES", "CHARGER" ];
g_VAR_ABBREV_LOOKUP = { 'P':"PHONE", 'W':"WALKMAN", 'R':"RADIO", 'B':"BATTERIES", 'C':"CHARGER" };


#
#  As we get EVIDENCE input from the user, we collect it in this map of key-values.
#  Examples of evidence key-vals:
#      'RADIO':1        # means he bought a RADIO
#      'CHARGER':0    # means he did NOT buy a CHARGER
#      'PHONE':0.4      # we are 40% confident that he bought a PHONE (from some external evidence)
#
g_EVIDENCE_MAP = {}


g_DEBUG_ON = False

g_ENABLE_BACKWARD_PROPAGATION = False   # BACKWARD_PROPAGATION is initially disabled for the initial phase of this assignment


################################################################################
#
#  NegateVariableKey
#
#  Negate the meaning of our internal true/false variable key string.
#
#  e.g. converts "PHONE" to "!PHONE"  ; and "!PHONE" to "PHONE"
#
def NegateVariableKey(var_name):

  if var_name[0] == '!':
    var_name = var_name[1:]            # remove the "!"  ("not")
  else:
    var_name = "!" + var_name          # prepend a "!"    ("not")

  return var_name

################################################################################
################################################################################


################################################################################
#
# PrintCurrentEvidence
#
def PrintCurrentEvidence():

  print(" ALL CURRENT (accumulated) EVIDENCE:")

  for var_key in g_EVIDENCE_MAP:
    val = g_EVIDENCE_MAP[var_key]
    val_display_str = "TRUE" if (val == 1.0) else "FALSE" if (val == 0.0) else ("%0.04f" % val)
    print("     %s : %s " % (var_key, val_display_str))

################################################################################
################################################################################


################################################################################
#
#  PrintCurrentProbabilities
#
def PrintCurrentProbabilities(var_prob_map):

  print(" ALL VARIABLE PROBABILITIES:")

  for var_key in var_prob_map:
    var_prob = var_prob_map[var_key]
    print("     %s = %0.04f " % (var_key, var_prob))
  
################################################################################
################################################################################


################################################################################
#
#  GetAllInputVariables
#
#  Return a list of all input variables for a given variable.
#  e.g. for "WALKMAN", return "PHONE"; optionally, also include "!PHONE"
#
def GetAllInputVariables(var_name, include_neg_vars=False):
  input_vars_map = {}
  
  if var_name in g_COND_PROBS:
    prob_keys_map = g_COND_PROBS[var_name]

    for compound_var_key in prob_keys_map:      # compound_var_key is like e.g.  "PHONE,!RADIO,WALKMAN"  from inside g_COND_PROBS{} .
      pos_neg_keys_list = compound_var_key.split(',') if compound_var_key else []    # split into e.g. [ "PHONE", "!RADIO", "WALKMAN" ]
      for pos_neg_key in pos_neg_keys_list:
        if (pos_neg_key[0] != '!') or include_neg_vars:
          input_vars_map[pos_neg_key] = 1


  input_vars_list = input_vars_map.keys()

  return input_vars_list

################################################################################
################################################################################


################################################################################
#
#  GetAllOutputVariables
#
#  Return a list of all OUTPUT variables for a given variable.
#  e.g.  for "PHONE", return "WALKMAN"
#
def GetAllOutputVariables(var_name):
  output_vars_list = []

  #
  #  Walk all the OTHER variables
  #
  for other_var_key in g_ALL_VARIABLES:
    if (other_var_key != var_name) and not (other_var_key in output_vars_list) :
      #
      # Get the INPUT variables for this other variable.
      #
      input_vars_list = GetAllInputVariables(other_var_key)
      if var_name in input_vars_list:
        #
        # var_name is an INPUT for other_var_key.  That means other_var_key is an OUTPUT for var_name. 
        #
        output_vars_list.append(other_var_key)
  
  return output_vars_list

################################################################################
################################################################################


################################################################################
#
#  InferVarProbFromInputs
#
#  var_name is the name of a variable.
#  All its INPUTS appear in all_input_vars_list[].
#  var_prob_map is a dict (associative array) with values for each input variable.
#
#  Return the probability of variable [var_name],
#  given the provided neighbor-variable probabilities, and the conditional probabilities in g_COND_PROBS[var_name].
#
def InferVarProbFromInputs(var_name, all_input_vars_list, var_prob_map):
  var_prob = 0.0

  if g_DEBUG_ON:
    print(" > InferVarProbFromInputs  %s %s %s  " % (var_name, str(all_input_vars_list), str(var_prob_map)))

  prob_keys_map = g_COND_PROBS[var_name]

  for compound_var_key in prob_keys_map:      # compound_var_key is like e.g.  "PHONE,!RADIO,WALKMAN"  from inside g_COND_PROBS{} .
    cond_prob = prob_keys_map[compound_var_key]

    if compound_var_key == "":
      #
      # They condition key is EMPTY.  That means this is a ROOT VARIABLE, with NO INPUTS.  So just use the given PRIOR probability.
      #
      if (var_name in var_prob_map) and (var_prob_map[var_name] >= 0.0):
        #
        # Don't overwrite an existing legal value in the working var_prob_map, because it may be a product of back-propagation.
        # Just echo the current value.
        #
        var_prob = var_prob_map[var_name]
      else:
        var_prob = cond_prob
      break
      
    else:
    
      pos_neg_keys_list = compound_var_key.split(',') if compound_var_key else []    # split into e.g. [ "PHONE", "!RADIO", "WALKMAN" ]


      ##############################
      #
      #  COMPLETE THIS CODE  # 1
      #
      ##############################




  if g_DEBUG_ON:
    print(" < InferVarProbFromInputs %s  returning %0.04f " % (var_name, var_prob))
  
  return var_prob

################################################################################
################################################################################


################################################################################
#
#  InferVarProbFromOutputs
#
#  var_name is the name of a variable.
#  All its OUTPUTS appear in all_output_vars_list[].
#  var_prob_map is a dict (associative array) with values for ALL variables (since we've already run forward-propagation).
#
#  Return the probability of variable [var_name],
#  given the provided neighbor-variable probabilities, and the conditional probabilities in g_COND_PROBS[].
#
def InferVarProbFromOutputs(var_name, all_output_vars_list, var_prob_map):
  var_prob = 0.0

  if g_DEBUG_ON:
    print(" > InferVarProbFromOutputs  %s %s %s  " % (var_name, str(all_output_vars_list), str(var_prob_map)))


  ##############################
  #
  #  COMPLETE THIS CODE  # 2
  #
  #  This one is much harder, and you don't have to implement it in the first phase of the assignment; unless you're adventurous.
  #  If you do, you'll have to set g_ENABLE_BACKWARD_PROPAGATION = True  at top, to test this.
  #
  ##############################



  if g_DEBUG_ON:
    print(" < InferVarProbFromOutputs %s  returning %0.04f " % (var_name, var_prob))

  return var_prob

################################################################################
################################################################################


  
################################################################################
#
#  RunInference
#
def RunInference():

  var_prob_map = {}

  try:
  
    #
    #  Initialize a probabilities map, setting each one initially to an illegal value.
    #
    for var_key in g_ALL_VARIABLES:
      var_prob_map[var_key] = -1

    #
    #  Keep track of which variables are stabilized; and how many unstabilized ones remain.
    #
    stabilized_vars_map = {}
    remaining_vars = len(g_ALL_VARIABLES)


    #
    #  1.  Process EVIDENCE input by the user. 
    #
    for var_key in g_EVIDENCE_MAP:
      #
      #  The USER has input EVIDENCE for this variable.  So just plug that value in.
      #
      var_prob_map[var_key] = g_EVIDENCE_MAP[var_key]
      stabilized_vars_map[var_key] = 1
      remaining_vars -= 1



    ##########################################
    #
    #  REPEAT FORWARD and BACKWARD PROPAGATION.
    #  Usually we would do this until the network stabilizes (within some smoothing to discount rounding jitter).
    #  Here we'll just artificially limit the total outer loops to 4.
    #
    outer_loops_count = 0
    outer_making_progress = True
    while remaining_vars and outer_making_progress and (outer_loops_count < 4):
      #
      #  We must make PROGRESS on each loop ...
      #
      outer_making_progress = False
      outer_loops_count += 1

      if g_DEBUG_ON:
        print(" Running OUTER LOOP #%d ... " % outer_loops_count)



      ########################################
      #
      #  2.  LOOP, applying FORWARD PROPAGATION from INPUT VARIABLES, until we've found probabilities for all variables (or failed)
      #
      inner_making_progress = True
      while remaining_vars and inner_making_progress:
        #
        #  We must make PROGRESS on each loop ...
        #
        inner_making_progress = False

        if g_DEBUG_ON:
          print(" Running FORWARD PROPARATION ... ")
        
        #
        #  Find an unprocessed variable, for which we have all required inputs.
        #
        for var_key in var_prob_map:

          if var_key in g_EVIDENCE_MAP:
            #
            #  Don't modify evidence variables.
            #
            pass

          elif var_key in stabilized_vars_map:
            pass
            
          else:
            #
            #  Find all its INPUT variables.
            #
            all_input_vars_list = GetAllInputVariables(var_key, include_neg_vars=False)
            
            #
            # Do we have all its INPUTS ?
            #
            have_all_inputs = True
            for input_var in all_input_vars_list:
              if var_prob_map[input_var] == -1:
                # NOPE !  A requred input variable does not have a value yet.  Keep searching.
                have_all_inputs = False
                break
  
            if have_all_inputs:
              #
              # We have all input variables for unprocessed variable [var_key].  So we can now PROCESS IT.
              #
              new_prob = InferVarProbFromInputs(var_key, all_input_vars_list, var_prob_map)

              #
              #  Was the variable's probability updated ?
              #  Do an *approximate* EQUALS check (to defeat rounding errors).
              #
              is_roughly_equal = (abs(new_prob - var_prob_map[var_key]) < 0.000001)
              if is_roughly_equal:
                stabilized_vars_map[var_key] = 1
                remaining_vars -= 1
                if g_DEBUG_ON:
                  print(" (stabilized variable '%s') " % var_key)
              else:
                #
                #  We have successfully UPDATED this probability.  This is PROGRESS ...
                #
                var_prob_map[var_key] = new_prob
                inner_making_progress = True
                outer_making_progress = True


        if not inner_making_progress:
          break

      #
      #  END of FORWARD PROPAGATION LOOP
      #
      ########################################


      #
      #  We initially DISABLE BACKWARD PROPAGATION, so the algorithm will terminate without that algorithm implemented.
      #
      if not g_ENABLE_BACKWARD_PROPAGATION:
        pass
      else:
        # Reset for backward-propagation
        stabilized_vars_map = {}
        remaining_vars = len(g_ALL_VARIABLES)
  
        ########################################
        #
        #  3.  LOOP AGAIN, applying BACKWARD PROPAGATION from OUTPUT VARIABLES, until we've found probabilities for all variables (or failed)
        #
        inner_making_progress = True
        while remaining_vars and inner_making_progress:
          #
          #  We must make PROGRESS on each loop ...
          #
          inner_making_progress = False
  
          if g_DEBUG_ON:
            print(" Running BACKWARD PROPARATION ... ")
          
          #
          #  Find an unprocessed variable, for which we have all required inputs.
          #
          for var_key in var_prob_map:
      
            if var_key in g_EVIDENCE_MAP:
              #
              #  Don't modify evidence variables.
              #
              pass
  
            elif var_key in stabilized_vars_map:
              pass
  
            elif var_prob_map[var_key] == -1:
              #
              #  We ran forward-propagation before backward-propagation.
              #  So by now we should have a PRIOR probability for this variable.  Otherwise, we cannot proceed.
              #
              print(" ERROR: cannot run back-propagation for variable '%s', because there is no PRIOR. " % var_key)
              break
              
            else:
            
              #
              #  This is an unprocessed variable.  Find all its OUTPUT variables.
              #
              all_output_vars_list = GetAllOutputVariables(var_key)
              
              #
              # Do we have all its OUTPUTS ?
              #
              have_all_outputs = True
              for output_var in all_output_vars_list:
                if var_prob_map[output_var] == -1:
                  # NOPE !  An output variable does not have a value yet.
                  have_all_outputs = False
                  break
    
              if have_all_outputs:
                #
                #  We DON'T have all INPUT variables; but we have all OUTPUT variables.
                #  So we can BACK-PROPAGATE using [Naive] Bayes.
                #
                new_prob = InferVarProbFromOutputs(var_key, all_output_vars_list, var_prob_map)
  
                #
                #  Was the variable's probability updated ?
                #  Do an *approximate* EQUALS check (to defeat rounding errors).
                #
                is_roughly_equal = (abs(new_prob - var_prob_map[var_key]) < 0.000001)
                if is_roughly_equal:
                  stabilized_vars_map[var_key] = 1
                  remaining_vars -= 1
                  if g_DEBUG_ON:
                    print(" (stabilized variable '%s') " % var_key)
                else:
                  #
                  #  We have successfully UPDATED this probability.  This is PROGRESS ...
                  #
                  var_prob_map[var_key] = new_prob
                  inner_making_progress = True
                  outer_making_progress = True
  
  
          if not inner_making_progress:
            break
  
        #
        #  END of BACKWARD PROPAGATION LOOP
        #
        ########################################


      if not outer_making_progress:
        break

    #
    #  END of OUTER LOOP
    #
    ##########################################

          
  except Exception as e:
    print("EXCEPTION in RunInference() : %s " % str(e))


  if remaining_vars:
    print(" ERROR:  RunInference() has FAILED to produce a consistent set of probabilities.")
    var_prob_map = None


  return var_prob_map
  
################################################################################
################################################################################


################################################################################
#
#  RunQueryLoop
#
def RunQueryLoop():
  global g_EVIDENCE_MAP
  global g_DEBUG_ON

  #
  #  Print the initial state of the PROBABILITIES.
  #
  var_prob_map = RunInference()
  if var_prob_map:
    print("\r\n ------------------------------ ")
    PrintCurrentProbabilities(var_prob_map)
    print(" ------------------------------\r\n ")


  #
  #  Loop forever.  On each loop, query USER for INPUT, run INFERENCE, and print status.
  #
  while True:

    #
    # Query the user for EVIDENCE input ...
    #
    print("\r\n ENTER COMMAND:  (e.g. 'P' or '!P'  to set PHONE evidence variable true/false;\r\n                       or e.g. 'P=0.5' ; 'RESET' to clear evidence; or 'DEBUG')")

    #
    # Clean up user response, and convert to upper case.
    #
    try:
      response = raw_input(" >> ").upper().strip()
    except Exception as e:
      # user probably hit Control+C ; exit gracefully below
      response = "EXIT"
      
    show_update = True


    if response == "":
      # ignore empty response
      show_update = False
      pass
      
    elif response in [ "RESET", "CLEAR" ] :
      g_EVIDENCE_MAP = {}   # clear all evidence
      
    elif (response in g_VAR_ABBREV_LOOKUP) or (response in g_ALL_VARIABLES):
      #
      # Set the evidence variable to TRUE
      #
      var_name = response if (response in g_ALL_VARIABLES) else g_VAR_ABBREV_LOOKUP[response]
      g_EVIDENCE_MAP[var_name] = 1.0

    elif (response and (response[0] == '!')):
      #
      #  This looks like a NEGATED variable
      #
      pos_var = NegateVariableKey(response)  # get the positive part of the variable or abbreviation (e.g.  "!P" -> "P")
      if (pos_var in g_VAR_ABBREV_LOOKUP) or (pos_var in g_ALL_VARIABLES) :
        #
        # Set the evidence variable to FALSE
        #
        var_name = pos_var if (pos_var in g_ALL_VARIABLES) else g_VAR_ABBREV_LOOKUP[pos_var]
        g_EVIDENCE_MAP[var_name] = 0.0
      else:
        print(" ERROR: attempted negation of undefined variable '%s' " % pos_var)
        show_update = False

    elif ('=' in response):
      #
      #  This looks like user is setting a specific value for a variable;  e.g. "P=0.5"
      #
      parts = response.split('=')
      if len(parts) == 2:
        var_name = parts[0].strip()    #  e.g. "P" or "PHONE"
        val_str = parts[1].strip()         #  e.g.  "0.5"
        if var_name in g_VAR_ABBREV_LOOKUP:
          var_name = g_VAR_ABBREV_LOOKUP[var_name]    #  e.g. "P" -> "PHONE"
        if var_name in g_ALL_VARIABLES:
          # is the value a proper floating point value, in [0.0, 1.0] ?
          try:
            prob_val = float(val_str)
            if (prob_val >= 0.0) and (prob_val <= 1.0) :
              g_EVIDENCE_MAP[var_name] = prob_val
            else:
              print(" (bad value; must be a probability in [0.0, 1.0] )")
              show_update = False
          except:
            print(" (input did not parse as a floating point value; try e.g.  'P=0.5' )")
            show_update = False
        else:
          print(" (unrecognized variable)")
          show_update = False      
      else:
        print(" (bad format; try e.g.  'P=0.5' )")
        show_update = False

    elif response == "DEBUG":
      g_DEBUG_ON = not g_DEBUG_ON
      print(" DEBUG tracing is %s " % ("ON" if g_DEBUG_ON else "OFF"))
      show_update = False

      
    elif response in [ "BREAK", "EXIT", "QUIT" ] :
      show_update = False
      break

    else:
      all_abbrev_vars_list = g_VAR_ABBREV_LOOKUP.keys()
      neg_abbrev_vars_list = [ "!"+key for key in all_abbrev_vars_list]
      print("\r\n ERROR: undefined command; legal commands are:\r\n            %s %s ;\r\n            or e.g. 'P=0.5' ; 'RESET' ; or 'DEBUG' " % (all_abbrev_vars_list, neg_abbrev_vars_list))
      show_update = False


    #
    #  PRINT the current state (EVIDENCE and PROBABILITIES).
    #
    if show_update:
      #
      #  RUN INFERENCE, applying the new evidence.
      #
      var_prob_map = RunInference()
      print("\r\n ------------------------------ ")
      PrintCurrentEvidence()
      print("\r\n")
      if var_prob_map:
        PrintCurrentProbabilities(var_prob_map)
      print(" ------------------------------\r\n ")

  
################################################################################
################################################################################


################################################################################
#
# This is the main() entrypoint of every top-level Python program. 
#
if __name__ == '__main__':
  RunQueryLoop()

################################################################################
################################################################################

