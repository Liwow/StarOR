\ Model Generator_Optimization
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  4 x_0_Type0 + 4 x_0_Type1 + 3 x_0_Type2 + 4 x_1_Type0 + 4 x_1_Type1
   + 3 x_1_Type2 + 4 x_2_Type0 + 4 x_2_Type1 + 3 x_2_Type2 + 4 x_3_Type0
   + 4 x_3_Type1 + 3 x_3_Type2 + 4 x_4_Type0 + 4 x_4_Type1 + 3 x_4_Type2
   + 353 y_0_Type0 + 349 y_0_Type1 + 351 y_0_Type2 + 353 y_1_Type0
   + 349 y_1_Type1 + 351 y_1_Type2 + 353 y_2_Type0 + 349 y_2_Type1
   + 351 y_2_Type2 + 353 y_3_Type0 + 349 y_3_Type1 + 351 y_3_Type2
   + 353 y_4_Type0 + 349 y_4_Type1 + 351 y_4_Type2 + 80 z_0_Type0
   + 78 z_0_Type1 + 78 z_0_Type2 + 80 z_1_Type0 + 78 z_1_Type1
   + 78 z_1_Type2 + 80 z_2_Type0 + 78 z_2_Type1 + 78 z_2_Type2
   + 80 z_3_Type0 + 78 z_3_Type1 + 78 z_3_Type2 + 80 z_4_Type0
   + 78 z_4_Type1 + 78 z_4_Type2
Subject To
 z_change_1_Type0: - z_0_Type0 + z_1_Type0 - delta_z_1_Type0 = 0
 z_change_1_Type1: - z_0_Type1 + z_1_Type1 - delta_z_1_Type1 = 0
 z_change_1_Type2: - z_0_Type2 + z_1_Type2 - delta_z_1_Type2 = 0
 z_change_2_Type0: - z_1_Type0 + z_2_Type0 - delta_z_2_Type0 = 0
 z_change_2_Type1: - z_1_Type1 + z_2_Type1 - delta_z_2_Type1 = 0
 z_change_2_Type2: - z_1_Type2 + z_2_Type2 - delta_z_2_Type2 = 0
 z_change_3_Type0: - z_2_Type0 + z_3_Type0 - delta_z_3_Type0 = 0
 z_change_3_Type1: - z_2_Type1 + z_3_Type1 - delta_z_3_Type1 = 0
 z_change_3_Type2: - z_2_Type2 + z_3_Type2 - delta_z_3_Type2 = 0
 z_change_4_Type0: - z_3_Type0 + z_4_Type0 - delta_z_4_Type0 = 0
 z_change_4_Type1: - z_3_Type1 + z_4_Type1 - delta_z_4_Type1 = 0
 z_change_4_Type2: - z_3_Type2 + z_4_Type2 - delta_z_4_Type2 = 0
 start_up_0_Type0: 5 y_0_Type0 >= 0
 start_up_0_Type1: 2 y_0_Type1 >= 0
 start_up_0_Type2: y_0_Type2 >= 0
 start_up_1_Type0: - 5 y_1_Type0 + delta_z_1_Type0 <= 0
 start_up_1_Type1: - 2 y_1_Type1 + delta_z_1_Type1 <= 0
 start_up_1_Type2: - y_1_Type2 + delta_z_1_Type2 <= 0
 start_up_2_Type0: - 5 y_2_Type0 + delta_z_2_Type0 <= 0
 start_up_2_Type1: - 2 y_2_Type1 + delta_z_2_Type1 <= 0
 start_up_2_Type2: - y_2_Type2 + delta_z_2_Type2 <= 0
 start_up_3_Type0: - 5 y_3_Type0 + delta_z_3_Type0 <= 0
 start_up_3_Type1: - 2 y_3_Type1 + delta_z_3_Type1 <= 0
 start_up_3_Type2: - y_3_Type2 + delta_z_3_Type2 <= 0
 start_up_4_Type0: - 5 y_4_Type0 + delta_z_4_Type0 <= 0
 start_up_4_Type1: - 2 y_4_Type1 + delta_z_4_Type1 <= 0
 start_up_4_Type2: - y_4_Type2 + delta_z_4_Type2 <= 0
 z_lb_0_Type0: z_0_Type0 >= 0
 z_ub_0_Type0: z_0_Type0 <= 5
 z_lb_0_Type1: z_0_Type1 >= 0
 z_ub_0_Type1: z_0_Type1 <= 2
 z_lb_0_Type2: z_0_Type2 >= 0
 z_ub_0_Type2: z_0_Type2 <= 1
 z_lb_1_Type0: z_1_Type0 >= 0
 z_ub_1_Type0: z_1_Type0 <= 5
 z_lb_1_Type1: z_1_Type1 >= 0
 z_ub_1_Type1: z_1_Type1 <= 2
 z_lb_1_Type2: z_1_Type2 >= 0
 z_ub_1_Type2: z_1_Type2 <= 1
 z_lb_2_Type0: z_2_Type0 >= 0
 z_ub_2_Type0: z_2_Type0 <= 5
 z_lb_2_Type1: z_2_Type1 >= 0
 z_ub_2_Type1: z_2_Type1 <= 2
 z_lb_2_Type2: z_2_Type2 >= 0
 z_ub_2_Type2: z_2_Type2 <= 1
 z_lb_3_Type0: z_3_Type0 >= 0
 z_ub_3_Type0: z_3_Type0 <= 5
 z_lb_3_Type1: z_3_Type1 >= 0
 z_ub_3_Type1: z_3_Type1 <= 2
 z_lb_3_Type2: z_3_Type2 >= 0
 z_ub_3_Type2: z_3_Type2 <= 1
 z_lb_4_Type0: z_4_Type0 >= 0
 z_ub_4_Type0: z_4_Type0 <= 5
 z_lb_4_Type1: z_4_Type1 >= 0
 z_ub_4_Type1: z_4_Type1 <= 2
 z_lb_4_Type2: z_4_Type2 >= 0
 z_ub_4_Type2: z_4_Type2 <= 1
 supply_0: x_0_Type0 + x_0_Type1 + x_0_Type2 >= 163
 supply_1: x_1_Type0 + x_1_Type1 + x_1_Type2 >= 180
 supply_2: x_2_Type0 + x_2_Type1 + x_2_Type2 >= 208
 supply_3: x_3_Type0 + x_3_Type1 + x_3_Type2 >= 183
 supply_4: x_4_Type0 + x_4_Type1 + x_4_Type2 >= 227
 min_power_0_Type0: x_0_Type0 >= 35
 max_power_0_Type0: x_0_Type0 <= 126
 min_power_0_Type1: x_0_Type1 >= 35
 max_power_0_Type1: x_0_Type1 <= 130
 min_power_0_Type2: x_0_Type2 >= 35
 max_power_0_Type2: x_0_Type2 <= 125
 min_power_1_Type0: x_1_Type0 >= 35
 max_power_1_Type0: x_1_Type0 <= 126
 min_power_1_Type1: x_1_Type1 >= 35
 max_power_1_Type1: x_1_Type1 <= 130
 min_power_1_Type2: x_1_Type2 >= 35
 max_power_1_Type2: x_1_Type2 <= 125
 min_power_2_Type0: x_2_Type0 >= 35
 max_power_2_Type0: x_2_Type0 <= 126
 min_power_2_Type1: x_2_Type1 >= 35
 max_power_2_Type1: x_2_Type1 <= 130
 min_power_2_Type2: x_2_Type2 >= 35
 max_power_2_Type2: x_2_Type2 <= 125
 min_power_3_Type0: x_3_Type0 >= 35
 max_power_3_Type0: x_3_Type0 <= 126
 min_power_3_Type1: x_3_Type1 >= 35
 max_power_3_Type1: x_3_Type1 <= 130
 min_power_3_Type2: x_3_Type2 >= 35
 max_power_3_Type2: x_3_Type2 <= 125
 min_power_4_Type0: x_4_Type0 >= 35
 max_power_4_Type0: x_4_Type0 <= 126
 min_power_4_Type1: x_4_Type1 >= 35
 max_power_4_Type1: x_4_Type1 <= 130
 min_power_4_Type2: x_4_Type2 >= 35
 max_power_4_Type2: x_4_Type2 <= 125
 margin_0: x_0_Type0 + x_0_Type1 + x_0_Type2 >= 187.45
 margin_1: x_1_Type0 + x_1_Type1 + x_1_Type2 >= 207
 margin_2: x_2_Type0 + x_2_Type1 + x_2_Type2 >= 239.2
 margin_3: x_3_Type0 + x_3_Type1 + x_3_Type2 >= 210.45
 margin_4: x_4_Type0 + x_4_Type1 + x_4_Type2 >= 261.05
Bounds
 z_0_Type0 <= 5
 z_0_Type1 <= 2
 z_0_Type2 <= 1
 z_1_Type0 <= 5
 z_1_Type1 <= 2
 z_1_Type2 <= 1
 z_2_Type0 <= 5
 z_2_Type1 <= 2
 z_2_Type2 <= 1
 z_3_Type0 <= 5
 z_3_Type1 <= 2
 z_3_Type2 <= 1
 z_4_Type0 <= 5
 z_4_Type1 <= 2
 z_4_Type2 <= 1
 -5 <= delta_z_1_Type0 <= 5
 -2 <= delta_z_1_Type1 <= 2
 -1 <= delta_z_1_Type2 <= 1
 -5 <= delta_z_2_Type0 <= 5
 -2 <= delta_z_2_Type1 <= 2
 -1 <= delta_z_2_Type2 <= 1
 -5 <= delta_z_3_Type0 <= 5
 -2 <= delta_z_3_Type1 <= 2
 -1 <= delta_z_3_Type2 <= 1
 -5 <= delta_z_4_Type0 <= 5
 -2 <= delta_z_4_Type1 <= 2
 -1 <= delta_z_4_Type2 <= 1
Binaries
 y_0_Type0 y_0_Type1 y_0_Type2 y_1_Type0 y_1_Type1 y_1_Type2 y_2_Type0
 y_2_Type1 y_2_Type2 y_3_Type0 y_3_Type1 y_3_Type2 y_4_Type0 y_4_Type1
 y_4_Type2
Generals
 z_0_Type0 z_0_Type1 z_0_Type2 z_1_Type0 z_1_Type1 z_1_Type2 z_2_Type0
 z_2_Type1 z_2_Type2 z_3_Type0 z_3_Type1 z_3_Type2 z_4_Type0 z_4_Type1
 z_4_Type2 delta_z_1_Type0 delta_z_1_Type1 delta_z_1_Type2 delta_z_2_Type0
 delta_z_2_Type1 delta_z_2_Type2 delta_z_3_Type0 delta_z_3_Type1
 delta_z_3_Type2 delta_z_4_Type0 delta_z_4_Type1 delta_z_4_Type2
End
