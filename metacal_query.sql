SELECT
metacal.unsheared_coadd_object_id,
metacal.unsheared_ra, metacal.unsheared_dec,
metacal.unsheared_e_1, metacal.unsheared_e_2, metacal.unsheared_weight, 
metacal.unsheared_r11, metacal.unsheared_r12, metacal.unsheared_r21, metacal.unsheared_r22, 
metacal.unsheared_mask_frac,
--
metacal.sheared_1p_e_1, metacal.sheared_1p_e_2, metacal.sheared_1p_weight,
metacal.sheared_1m_e_1, metacal.sheared_1m_e_2, metacal.sheared_1m_weight,
metacal.sheared_2p_e_1, metacal.sheared_2p_e_2, metacal.sheared_2p_weight,
metacal.sheared_2m_e_1, metacal.sheared_2m_e_2, metacal.sheared_2m_weight,
--
gold.dnf_zmean_sof, gold.dnf_zmc_sof, gold.dnf_zsigma_sof
--
FROM des_metacal_v03_004 as metacal
INNER JOIN des_y3_gold_v2_2_c as gold ON metacal.unsheared_coadd_object_id = gold.coadd_object_id
WHERE metacal.unsheared_flags=0 
AND gold.flags_gold<8
AND metacal.unsheared_snr>10 
AND metacal.unsheared_snr<1000
