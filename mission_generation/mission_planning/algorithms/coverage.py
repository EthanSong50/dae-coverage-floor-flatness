import math
import cv2
import fields2cover as f2c

def mask_to_f2c_cells(mask):
    """
    OpenCV의 Contours를 사용해 이진 마스크(Binary Mask)에서 외곽선을 추출하고,
    이를 Fields2Cover(F2C) 연산에 필요한 Cell 객체로 변환함.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 다른 컨투어 선택 함수들(geometry.py의 estimate_min_width_px 등)과
    # 일관되게 최대 면적 컨투어를 사용함 - 노이즈로 생긴 작은 컨투어가
    # contours[0]으로 뽑히는 경우를 방지함.
    c = max(contours, key=cv2.contourArea)

    ring = f2c.LinearRing()
    # 외곽선 좌표를 F2C Polygon 형태(LinearRing)로 변환
    for pt in c:
        ring.addPoint(float(pt[0][0]), float(pt[0][1]))
    # 폐곡선을 만들기 위해 시작점 다시 추가
    ring.addPoint(float(c[0][0][0]), float(c[0][0][1]))
    
    cell = f2c.Cell()
    cell.addRing(ring)
    
    cells = f2c.Cells()
    cells.addGeometry(cell)
    return cells

def generate_raw_swaths(mask, robot_params, decompose=False, split_angle_rad=None):
    """
    decompose=True면, mask 폴리곤을 문지방 경계마다 볼록 조각으로 분할
    (Boustrophedon Decomposition)한 뒤, 가장 큰 조각(본체)만 골라 로봇의
    실제 물리 폭(width_px) 기준으로 중심선 스와스 1개를 생성함.

    문지방 돌출부(작은 조각들)는 F2C에 넘기지 않음 - Leg2(문지방 중심점 ->
    coverage 시작점)가 이미 record_pcd=True로 그 위를 지나가므로 F2C가 별도로
    커버할 필요가 없음(node_002 사례로 검증됨).

    swath_width_px(라이다 측정반경 기반) 방식은 wide 노드 및 decompose=False일
    때 그대로 유지함 - 그 용도(넓은 방에서 평행선 간격)엔 원래도 맞는 값임.
    op_width에 이 값을 그대로 쓰면 narrow/ultra_narrow 노드에서 통로 폭보다
    커져 F2C가 스와스를 못 만드는 문제가 있었음(HISTORY.md §2 참고) -
    narrow/ultra_narrow는 로봇 실제 물리 폭을 대신 씀.
    """
    f2c_cells = mask_to_f2c_cells(mask)
    if f2c_cells is None:
        return []

    if decompose:
        decomp = f2c.DECOMP_Boustrophedon()
        decomp.setSplitAngle(split_angle_rad if split_angle_rad is not None else 0.0)
        decomp_cells = decomp.decompose(f2c_cells)

        const_hl = f2c.HG_Const_gen()
        decomp_cells = const_hl.generateHeadlands(decomp_cells, 0.0)

        if decomp_cells.size() == 0:
            return []

        areas = [decomp_cells.getGeometry(i).area() for i in range(decomp_cells.size())]
        big_idx = areas.index(max(areas))

        main_body_cells = f2c.Cells()
        main_body_cells.addGeometry(decomp_cells.getGeometry(big_idx))

        op_width = robot_params['width_px']
        target_cells = main_body_cells
    else:
        op_width = robot_params['swath_width_px']
        target_cells = f2c_cells

    robot = f2c.Robot(op_width, op_width)
    swath_gen = f2c.SG_BruteForce()

    if split_angle_rad is not None and decompose:
        raw_swaths_by_cells = swath_gen.generateSwaths(split_angle_rad, robot.getWidth(), target_cells)
    else:
        swath_gen.setStepAngle(math.pi / 36.0)
        obj = f2c.OBJ_SwathLength()
        raw_swaths_by_cells = swath_gen.generateBestSwaths(obj, robot.getWidth(), target_cells)

    swath_pairs = []
    n_cells = target_cells.size()
    for cell_idx in range(n_cells):
        try: cell_swaths = raw_swaths_by_cells[cell_idx]
        except Exception:
            try: cell_swaths = raw_swaths_by_cells.at(cell_idx)
            except Exception: continue

        n_swaths = cell_swaths.size()
        if n_swaths == 0:
            continue

        # [핵심] decompose 모드에서는 여러 줄(narrow 폭 대비 촘촘한 간격) 중
        # 가운데 하나만 골라 왕복 없이 중심선 하나로 지나감 - 이동하면서
        # 사각지대가 메워진다는 전제(assist_mask/boundary_repass 로직)와 일관됨.
        indices_to_use = [n_swaths // 2] if decompose else range(n_swaths)

        for i in indices_to_use:
            try: sw = cell_swaths[i]
            except Exception: sw = cell_swaths.at(i)
            p1 = (int(sw.startPoint().getX()), int(sw.startPoint().getY()))
            p2 = (int(sw.endPoint().getX()), int(sw.endPoint().getY()))
            swath_pairs.append((p1, p2))

    return swath_pairs
